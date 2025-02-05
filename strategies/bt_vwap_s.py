import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover

class VWAPStrategy(Strategy):
    # Define parameters as class variables
    window = 10  # VWAP window
    ma_period = 5  # Short MA period
    risk_factor = 0.02  # Risk per trade
    drift_factor = 0.015  # Price drift expectation
    num_bins = 15  # Number of execution bins
    trade_threshold = 0.02  # Trade signal threshold

    def init(self):
        # Set instance parameters from class parameters or defaults
        self.window = getattr(self, 'window', 10)
        self.ma_period = getattr(self, 'ma_period', 5)
        self.risk_factor = getattr(self, 'risk_factor', 0.02)
        self.drift_factor = getattr(self, 'drift_factor', 0.015)
        self.num_bins = getattr(self, 'num_bins', 15)
        self.trade_threshold = getattr(self, 'trade_threshold', 0.02)

        # Calculate VWAP
        def calculate_vwap(price, volume, window):
            """Compute VWAP with a rolling window."""
            typical_price = price
            values = typical_price * volume
            vwap = pd.Series(np.cumsum(values) / np.cumsum(volume)).rolling(window).mean()
            return vwap.bfill()

        # Compute main indicators
        self.vwap = self.I(lambda: calculate_vwap(self.data.Close, self.data.Volume, self.window))
        self.short_ma = self.I(lambda x: pd.Series(x).rolling(self.ma_period).mean(), self.data.Close)
        
        # Calculate relative volume
        def calculate_relative_volume(volume, window):
            """Compute relative volume using a moving average."""
            cumulative_volume = np.cumsum(volume) / np.sum(volume)
            return pd.Series(cumulative_volume).rolling(window).mean().bfill()
        
        self.relative_volume = self.I(lambda: calculate_relative_volume(self.data.Volume, self.window))

    def next(self):
        # Calculate position size based on risk
        current_price = self.data.Close[-1]
        risk_amount = self.equity * self.risk_factor
        position_size = risk_amount / current_price
        # Round to whole number of units and ensure minimum size
        position_size = max(1, round(position_size))

        # Price Momentum Check
        price_momentum = self.data.Close[-1] - self.short_ma[-1]
        
        # Volume confirmation
        volume_confirmation = self.data.Volume[-1] > self.data.Volume[-2]

        # Buy when price is below VWAP with strong upward momentum and high volume
        if (self.data.Close[-1] < self.vwap[-1] - self.trade_threshold and 
            price_momentum > 0 and volume_confirmation):
            self.buy(size=position_size)

        # Sell when price is above VWAP with strong downward momentum and high volume
        elif (self.data.Close[-1] > self.vwap[-1] + self.trade_threshold and 
              price_momentum < 0 and volume_confirmation):
            self.sell(size=position_size)

        # Exit conditions
        for trade in self.trades:
            # Exit long position if price crosses below VWAP
            if trade.is_long and crossover(self.vwap, self.data.Close):
                self.position.close()
            # Exit short position if price crosses above VWAP
            elif trade.is_short and crossover(self.data.Close, self.vwap):
                self.position.close() 