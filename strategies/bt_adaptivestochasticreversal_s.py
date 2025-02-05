import pandas as pd
import numpy as np
from backtesting import Strategy

class AdaptiveStochasticReversalStrategy(Strategy):
    """
    A strategy that uses Stochastic RSI with adaptive thresholds for different timeframes.
    Uses pandas calculations instead of talib for broader compatibility.
    """
    
    # Define parameters as class variables first
    stoch_rsi_period = 14
    stoch_rsi_smoothK = 3
    stoch_rsi_smoothD = 3
    weekly_oversold = 15
    weekly_overbought = 80
    shorter_timeframe_oversold = 20
    shorter_timeframe_overbought = 80
    risk_pct = 0.02
    rr_ratio = 2.0
    
    # Parameter configuration for optimization
    param_config = {
        "stoch_rsi_period": {
            "default": 14,
            "range": [10, 14, 20, 25]
        },
        "stoch_rsi_smoothK": {
            "default": 3,
            "range": [2, 3, 4, 5]
        },
        "stoch_rsi_smoothD": {
            "default": 3,
            "range": [2, 3, 4, 5]
        },
        "weekly_oversold": {
            "default": 15,
            "range": [10, 15, 20]
        },
        "weekly_overbought": {
            "default": 80,
            "range": [75, 80, 85]
        },
        "shorter_timeframe_oversold": {
            "default": 20,
            "range": [15, 20, 25]
        },
        "shorter_timeframe_overbought": {
            "default": 80,
            "range": [75, 80, 85]
        },
        "risk_pct": {
            "default": 0.02,
            "range": [0.01, 0.02, 0.03]
        },
        "rr_ratio": {
            "default": 2.0,
            "range": [1.5, 2.0, 2.5, 3.0]
        }
    }

    def init(self):
        # Calculate RSI
        def calculate_rsi(prices, period=14):
            # Convert to pandas Series and calculate price changes
            prices = pd.Series(prices)
            deltas = prices.diff()
            
            # Separate gains and losses
            gains = deltas.where(deltas > 0, 0.0)
            losses = -deltas.where(deltas < 0, 0.0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=period, min_periods=1).mean()
            avg_losses = losses.rolling(window=period, min_periods=1).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses.replace(0, np.nan)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))
            
            # Fill NaN values with 50 (neutral)
            rsi = rsi.fillna(50)
            
            return rsi

        # Calculate Stochastic RSI
        def calculate_stoch_rsi(prices, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
            # Calculate RSI
            rsi = calculate_rsi(prices, rsi_period)
            
            # Calculate Stochastic RSI
            rsi_min = rsi.rolling(window=stoch_period, min_periods=1).min()
            rsi_max = rsi.rolling(window=stoch_period, min_periods=1).max()
            
            # Avoid division by zero
            denominator = (rsi_max - rsi_min)
            stoch_k = np.where(denominator != 0, 
                             ((rsi - rsi_min) / denominator) * 100,
                             50)  # Use 50 as neutral value when denominator is 0
            
            # Convert to pandas Series for smoothing
            stoch_k = pd.Series(stoch_k).rolling(window=k_period, min_periods=1).mean()
            stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()
            
            return stoch_k, stoch_d

        # Calculate indicators using the wrapper
        prices = self.data.Close
        self.stoch_k, self.stoch_d = self.I(lambda: calculate_stoch_rsi(
            prices,
            rsi_period=self.stoch_rsi_period,
            stoch_period=self.stoch_rsi_period,
            k_period=self.stoch_rsi_smoothK,
            d_period=self.stoch_rsi_smoothD
        ), name='stoch_rsi')

    def next(self):
        if len(self.data) < self.stoch_rsi_period + max(self.stoch_rsi_smoothK, self.stoch_rsi_smoothD):
            return

        price = self.data.Close[-1]
        
        # Entry Logic: Look for oversold/overbought conditions
        if not self.position:
            # Long setup on weekly oversold
            if self.stoch_k[-1] < self.weekly_oversold:
                # Calculate position size based on risk
                stop_loss = price * 0.98  # 2% stop loss
                risk_per_unit = price - stop_loss
                
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    position_size = int(risk_amount / risk_per_unit)
                    if position_size > 0:
                        target = price + (risk_per_unit * self.rr_ratio)
                        self.buy(size=position_size, sl=stop_loss, tp=target)
            
            # Short setup on shorter timeframe overbought
            elif self.stoch_k[-1] > self.shorter_timeframe_overbought:
                # Calculate position size based on risk
                stop_loss = price * 1.02  # 2% stop loss
                risk_per_unit = stop_loss - price
                
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    position_size = int(risk_amount / risk_per_unit)
                    if position_size > 0:
                        target = price - (risk_per_unit * self.rr_ratio)
                        self.sell(size=position_size, sl=stop_loss, tp=target)

        # Exit Logic
        elif self.position:
            # Exit long position if overbought
            if self.position.is_long and self.stoch_k[-1] > self.weekly_overbought:
                self.position.close()
            
            # Exit short position if oversold
            elif self.position.is_short and self.stoch_k[-1] < self.shorter_timeframe_oversold:
                self.position.close() 