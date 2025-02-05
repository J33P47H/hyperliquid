import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

def custom_RSI(prices, period=14):
    """Calculate RSI using a custom implementation"""
    prices = np.array(prices)
    delta = np.diff(prices, prepend=prices[0])
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices, dtype=float)
    avg_loss = np.zeros_like(prices, dtype=float)
    
    # Calculate the first average gain and loss
    avg_gain[period - 1] = np.mean(gain[1:period+1])
    avg_loss[period - 1] = np.mean(loss[1:period+1])
    
    # Compute the smoothed averages
    for i in range(period, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
        
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

class RsiVwapStrategy(Strategy):
    # Define parameters as class variables
    rsi_period = 8
    oversold_threshold = 40
    overbought_threshold = 60
    risk_factor = 0.02
    
    def init(self):
        # Set instance parameters from class parameters or defaults
        self.rsi_period = getattr(self, 'rsi_period', 8)
        self.oversold_threshold = getattr(self, 'oversold_threshold', 40)
        self.overbought_threshold = getattr(self, 'overbought_threshold', 60)
        self.risk_factor = getattr(self, 'risk_factor', 0.02)
        
        # Calculate RSI
        self.rsi = self.I(custom_RSI, self.data.Close, self.rsi_period)
        
        # Calculate VWAP
        def compute_vwap(opens, highs, lows, closes, volumes):
            typical_price = (highs + lows + closes) / 3
            cumulative_vwap = (typical_price * volumes).cumsum() / volumes.cumsum()
            return cumulative_vwap

        self.vwap = self.I(
            compute_vwap,
            self.data.Open,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.data.Volume
        )

    def next(self):
        # Calculate position size based on risk
        current_price = self.data.Close[-1]
        risk_amount = self.equity * self.risk_factor
        position_size = risk_amount / current_price
        # Round to whole number of units and ensure minimum size
        position_size = max(1, round(position_size))

        # Long entry: RSI crosses above oversold and price crosses above VWAP
        if (not self.position and 
            self.rsi[-2] < self.oversold_threshold and 
            self.rsi[-1] >= self.oversold_threshold and 
            crossover(self.data.Close, self.vwap)):
            self.buy(size=position_size)

        # Short entry: RSI crosses below overbought and price crosses below VWAP
        elif (not self.position and 
              self.rsi[-2] > self.overbought_threshold and 
              self.rsi[-1] <= self.overbought_threshold and 
              crossover(self.vwap, self.data.Close)):
            self.sell(size=position_size)

        # Exit long position
        elif self.position.is_long:
            if (self.data.Close[-1] < self.vwap[-1] or 
                self.rsi[-1] > self.overbought_threshold):
                self.position.close()

        # Exit short position
        elif self.position.is_short:
            if (self.data.Close[-1] > self.vwap[-1] or 
                self.rsi[-1] < self.oversold_threshold):
                self.position.close() 