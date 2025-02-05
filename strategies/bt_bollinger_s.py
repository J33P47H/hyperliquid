import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover

class BollingerStrategy(Strategy):
    # Define parameters as class variables
    n1 = 20  # period for the SMA
    n2 = 50  # period for the trend filter SMA
    n_std_dev = 2  # number of standard deviations for bands
    risk_factor = 0.02  # risk per trade

    def init(self):
        # Set instance parameters from class parameters or defaults
        self.n1 = getattr(self, 'n1', 20)
        self.n2 = getattr(self, 'n2', 50)
        self.n_std_dev = getattr(self, 'n_std_dev', 2)
        self.risk_factor = getattr(self, 'risk_factor', 0.02)

        # Calculate SMA
        self.sma = self.I(lambda x: pd.Series(x).rolling(self.n1).mean(), self.data.Close)
        
        # Calculate Bollinger Bands
        def bollinger_band(data, n, std_dev, upper=True):
            sma = pd.Series(data).rolling(n).mean()
            std = pd.Series(data).rolling(n).std()
            return sma + (std * std_dev if upper else -std * std_dev)

        self.upper_band = self.I(lambda x: bollinger_band(x, self.n1, self.n_std_dev, True), self.data.Close)
        self.lower_band = self.I(lambda x: bollinger_band(x, self.n1, self.n_std_dev, False), self.data.Close)
        
        # Calculate trend filter SMA
        self.trend_sma = self.I(lambda x: pd.Series(x).rolling(self.n2).mean(), self.data.Close)

    def next(self):
        # Calculate position size based on risk
        current_price = self.data.Close[-1]
        risk_amount = self.equity * self.risk_factor
        position_size = risk_amount / current_price
        # Round to whole number of units and ensure minimum size
        position_size = max(1, round(position_size))

        # Long entry: Price crosses above lower band and is above trend SMA
        if (crossover(self.data.Close, self.lower_band) and 
            self.data.Close[-1] > self.trend_sma[-1]):
            self.buy(size=position_size)
            
        # Short entry: Price crosses below upper band and is below trend SMA
        elif (crossover(self.upper_band, self.data.Close) and 
              self.data.Close[-1] < self.trend_sma[-1]):
            self.sell(size=position_size)

        # Exit conditions
        for trade in self.trades:
            if trade.is_long and crossover(self.sma, self.data.Close):
                self.position.close()
            elif trade.is_short and crossover(self.data.Close, self.sma):
                self.position.close() 