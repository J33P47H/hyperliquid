import pandas as pd
import numpy as np
from backtesting import Strategy

class AdaptiveStochasticReversalStrategy(Strategy):
    """
    A strategy that uses Stochastic RSI with adaptive thresholds for different timeframes.
    Uses pandas calculations instead of talib for broader compatibility.
    Position sizing is removed; orders are executed with the default trade size.
    """
    
    # Define parameters as class variables
    stoch_rsi_period = 14
    stoch_rsi_smoothK = 3
    stoch_rsi_smoothD = 3
    weekly_oversold = 15
    weekly_overbought = 80
    shorter_timeframe_oversold = 20
    shorter_timeframe_overbought = 80
    rr_ratio = 2.0  # Risk-reward ratio for target calculation

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
        "rr_ratio": {
            "default": 2.0,
            "range": [1.5, 2.0, 2.5, 3.0]
        }
    }

    def init(self):
        # Calculate RSI
        def calculate_rsi(prices, period=14):
            prices = pd.Series(prices)
            deltas = prices.diff()
            
            gains = deltas.where(deltas > 0, 0.0)
            losses = -deltas.where(deltas < 0, 0.0)
            
            avg_gains = gains.rolling(window=period, min_periods=1).mean()
            avg_losses = losses.rolling(window=period, min_periods=1).mean()
            
            rs = avg_gains / avg_losses.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # Use 50 as a neutral value

        # Calculate Stochastic RSI
        def calculate_stoch_rsi(prices, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
            rsi = calculate_rsi(prices, rsi_period)
            
            rsi_min = rsi.rolling(window=stoch_period, min_periods=1).min()
            rsi_max = rsi.rolling(window=stoch_period, min_periods=1).max()
            
            denominator = (rsi_max - rsi_min)
            stoch_k = np.where(denominator != 0, ((rsi - rsi_min) / denominator) * 100, 50)
            
            stoch_k = pd.Series(stoch_k).rolling(window=k_period, min_periods=1).mean()
            stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()
            
            return stoch_k, stoch_d

        # Calculate the indicators and register them with the backtesting engine
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
            # Long setup on weekly oversold condition
            if self.stoch_k[-1] < self.weekly_oversold:
                stop_loss = price * 0.98  # 2% below entry for stop loss
                risk_per_unit = price - stop_loss
                if risk_per_unit > 0:
                    target = price + (risk_per_unit * self.rr_ratio)
                    self.buy(sl=stop_loss, tp=target)
            
            # Short setup on shorter timeframe overbought condition
            elif self.stoch_k[-1] > self.shorter_timeframe_overbought:
                stop_loss = price * 1.02  # 2% above entry for stop loss
                risk_per_unit = stop_loss - price
                if risk_per_unit > 0:
                    target = price - (risk_per_unit * self.rr_ratio)
                    self.sell(sl=stop_loss, tp=target)

        # Exit Logic: Close positions based on reverse signals
        elif self.position:
            if self.position.is_long and self.stoch_k[-1] > self.weekly_overbought:
                self.position.close()
            elif self.position.is_short and self.stoch_k[-1] < self.shorter_timeframe_oversold:
                self.position.close()
