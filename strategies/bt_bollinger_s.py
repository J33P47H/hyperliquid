import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover

class BollingerStrategy(Strategy):
    # Define parameters as class variables
    n1 = 20         # Period for the SMA & Bollinger Bands
    n2 = 50         # Period for the trend filter SMA
    n_std_dev = 2   # Number of standard deviations for the bands
    
    # Add missing parameters as class variables
    atr_period = 14
    atr_exit_mult = 1.5
    atr_threshold = 0.0
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
    adx_period = 14
    adx_max = 25
    min_volume_ratio = 1.0
    profit_target_rr = 2.0

    # Optimization parameter configuration
    param_config = {
        "n1": {"default": 20, "range": [10, 20, 30, 40]},
        "n2": {"default": 50, "range": [30, 50, 70, 100]},
        "n_std_dev": {"default": 2, "range": [1.5, 2, 2.5, 3]},
        "atr_period": {"default": 14, "range": [10, 14, 20]},
        "atr_exit_mult": {"default": 1.5, "range": [1.0, 1.5, 2.0, 2.5]},
        "atr_threshold": {"default": 0.0, "range": [0.0, 0.1, 0.2, 0.3]},
        "rsi_period": {"default": 14, "range": [10, 14, 20]},
        "rsi_overbought": {"default": 70, "range": [60, 70, 80]},
        "rsi_oversold": {"default": 30, "range": [20, 30, 40]},
        "adx_period": {"default": 14, "range": [10, 14, 20]},
        "adx_max": {"default": 25, "range": [20, 25, 30]},
        "min_volume_ratio": {"default": 1.0, "range": [0.8, 1.0, 1.2]},
        "profit_target_rr": {"default": 2.0, "range": [1.5, 2.0, 2.5, 3.0]}
    }
    def init(self):
        # (Optional) Override instance parameters if needed.
        self.n1 = getattr(self, 'n1', 20)
        self.n2 = getattr(self, 'n2', 50)
        self.n_std_dev = getattr(self, 'n_std_dev', 2)

        # Calculate the fast SMA (used in Bollinger Bands and as an exit signal)
        self.sma = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), self.data.Close)

        # Define a helper function to compute Bollinger Bands.
        def bollinger_band(data, n, std_dev, upper=True):
            sma = pd.Series(data).rolling(n).mean()
            std = pd.Series(data).rolling(n).std()
            # Return upper or lower band based on the flag
            return sma + (std * std_dev if upper else -std * std_dev)

        # Calculate Bollinger Bands
        self.upper_band = self.I(
            lambda x: bollinger_band(x, self.n1, self.n_std_dev, True), self.data.Close)
        self.lower_band = self.I(
            lambda x: bollinger_band(x, self.n1, self.n_std_dev, False), self.data.Close)
        
        # Calculate trend filter SMA (using a longer period)
        self.trend_sma = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), self.data.Close)

    def next(self):
        # Entry signals: Only enter if no open position exists.
        if not self.position:
            # Long entry: price crosses above lower band and is above trend SMA
            if (crossover(self.data.Close, self.lower_band) and 
                self.data.Close[-1] > self.trend_sma[-1]):
                self.buy()

            # Short entry: price crosses below upper band and is below trend SMA
            elif (crossover(self.upper_band, self.data.Close) and 
                  self.data.Close[-1] < self.trend_sma[-1]):
                self.sell()

        # Exit signals: Close the current position based on the SMA crossover.
        if self.position:
            # For a long position, exit if the price falls below the SMA.
            if self.position.is_long and crossover(self.sma, self.data.Close):
                self.position.close()
            # For a short position, exit if the price rises above the SMA.
            elif self.position.is_short and crossover(self.data.Close, self.sma):
                self.position.close()
