import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover

class StoicReversalStrategy(Strategy):
    """
    A disciplined trading strategy that uses Stochastic RSI to identify oversold and overbought conditions.
    This version uses pandas calculations instead of talib for broader compatibility.
    """
    
    # Define parameters as class variables first
    stoch_timeperiod = 14
    stoch_fastk_period = 3
    stoch_fastd_period = 3
    stoch_oversold = 20
    stoch_overbought = 80
    risk_pct = 0.01
    stop_loss_pct = 0.02
    risk_reward_ratio = 2.0
    
    # Parameter configuration for optimization
    param_config = {
        "stoch_timeperiod": {
            "default": 14,
            "range": [7, 9, 11, 13, 14]
        },
        "stoch_fastk_period": {
            "default": 3,
            "range": [2, 3, 4, 5]
        },
        "stoch_fastd_period": {
            "default": 3,
            "range": [2, 3, 4, 5]
        },
        "stoch_oversold": {
            "default": 20,
            "range": [10, 15, 20, 25, 30]
        },
        "stoch_overbought": {
            "default": 80,
            "range": [70, 75, 80, 85, 90]
        },
        "risk_pct": {
            "default": 0.01,
            "range": [0.005, 0.01, 0.015, 0.02]
        },
        "stop_loss_pct": {
            "default": 0.02,
            "range": [0.01, 0.015, 0.02, 0.025, 0.03]
        },
        "risk_reward_ratio": {
            "default": 2.0,
            "range": [1.5, 2.0, 2.5, 3.0]
        }
    }

    def init(self):
        # Calculate RSI
        def calculate_rsi(prices, period=14):
            deltas = np.diff(prices)
            seed = deltas[:period+1]
            up = seed[seed >= 0].sum()/period
            down = -seed[seed < 0].sum()/period
            rs = up/down
            rsi = np.zeros_like(prices)
            rsi[:period] = 100. - 100./(1. + rs)

            for i in range(period, len(prices)):
                delta = deltas[i - 1]
                if delta > 0:
                    upval = delta
                    downval = 0.
                else:
                    upval = 0.
                    downval = -delta

                up = (up * (period - 1) + upval) / period
                down = (down * (period - 1) + downval) / period
                rs = up/down
                rsi[i] = 100. - 100./(1. + rs)
            
            return rsi

        # Calculate Stochastic RSI
        def calculate_stoch_rsi(close_prices, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
            # Calculate RSI
            rsi = calculate_rsi(close_prices, rsi_period)
            
            # Calculate Stochastic RSI
            rsi_series = pd.Series(rsi)
            min_rsi = rsi_series.rolling(window=stoch_period).min()
            max_rsi = rsi_series.rolling(window=stoch_period).max()
            
            # Calculate %K
            stoch_k = 100 * (rsi_series - min_rsi) / (max_rsi - min_rsi)
            
            # Calculate %D (SMA of %K)
            stoch_d = stoch_k.rolling(window=d_period).mean()
            
            return stoch_k, stoch_d

        # Calculate indicators using the wrapper
        prices = self.data.Close
        self.stoch_k, self.stoch_d = self.I(lambda: calculate_stoch_rsi(
            prices,
            rsi_period=self.stoch_timeperiod,
            stoch_period=self.stoch_timeperiod,
            k_period=self.stoch_fastk_period,
            d_period=self.stoch_fastd_period
        ), name='stoch_rsi')

    def next(self):
        price = self.data.Close[-1]
        
        # Entry Logic: Look for oversold conditions
        if not self.position:
            if self.stoch_k[-1] < self.stoch_oversold:
                # Calculate risk-based position sizing
                stop_loss_price = price * (1 - self.stop_loss_pct)
                take_profit_price = price * (1 + self.risk_reward_ratio * self.stop_loss_pct)
                risk_amount = self.equity * self.risk_pct
                risk_per_unit = price - stop_loss_price
                
                if risk_per_unit > 0:
                    position_size = risk_amount / risk_per_unit
                    position_size = max(1, round(position_size))
                    self.buy(size=position_size, sl=stop_loss_price, tp=take_profit_price)

        # Exit Logic: Look for overbought conditions plus a %K/%D crossover
        elif self.position:
            if len(self.stoch_k) >= 2:
                # Check for crossover condition and overbought
                if (self.stoch_k[-2] < self.stoch_d[-2] and 
                    self.stoch_k[-1] > self.stoch_d[-1] and 
                    self.stoch_k[-1] > self.stoch_overbought):
                    self.position.close() 