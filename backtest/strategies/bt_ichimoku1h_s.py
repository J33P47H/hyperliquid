import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover

class Ichimoku1hStrategy(Strategy):
    # Class-level parameter configuration
    param_config = {
        "tenkan_period": {
            "default": 9,
            "range": [7, 9, 11, 13]
        },
        "kijun_period": {
            "default": 26,
            "range": [22, 26, 30, 34]
        },
        "senkou_span_b_period": {
            "default": 52,
            "range": [48, 52, 56, 60]
        },
        "displacement": {
            "default": 26,
            "range": [22, 26, 30]
        }
    }
    
    # Default parameter values
    tenkan_period = param_config["tenkan_period"]["default"]
    kijun_period = param_config["kijun_period"]["default"]
    senkou_span_b_period = param_config["senkou_span_b_period"]["default"]
    displacement = param_config["displacement"]["default"]

    def init(self):
        # Set instance parameters from class parameters or defaults
        self.tenkan_period = getattr(self, 'tenkan_period', 9)
        self.kijun_period = getattr(self, 'kijun_period', 26)
        self.senkou_span_b_period = getattr(self, 'senkou_span_b_period', 52)
        self.displacement = getattr(self, 'displacement', 26)

        # Calculate Ichimoku components
        def calculate_ichimoku_line(high, low, period):
            high_values = pd.Series(high).rolling(period).max()
            low_values = pd.Series(low).rolling(period).min()
            return (high_values + low_values) / 2

        # Calculate main components
        self.tenkan_sen = self.I(calculate_ichimoku_line, self.data.High, self.data.Low, self.tenkan_period)
        self.kijun_sen = self.I(calculate_ichimoku_line, self.data.High, self.data.Low, self.kijun_period)

        # Calculate Senkou Spans
        senkou_span_a = (self.tenkan_sen + self.kijun_sen) / 2
        self.senkou_span_a = self.I(lambda x: np.roll(x, self.displacement), senkou_span_a)
        
        senkou_span_b_raw = calculate_ichimoku_line(self.data.High, self.data.Low, self.senkou_span_b_period)
        self.senkou_span_b = self.I(lambda x: np.roll(x, self.displacement), senkou_span_b_raw)
        
        # Calculate Chikou Span (shifted backwards)
        self.chikou_span = self.I(lambda x: np.roll(x, -self.displacement), self.data.Close)

    def next(self):
        # Wait for enough data
        required_bars = max(self.displacement, self.senkou_span_b_period)
        if len(self.data) < required_bars:
            return

        # Current price and indicators
        price = self.data.Close[-1]
        tenkan = self.tenkan_sen[-1]
        kijun = self.kijun_sen[-1]
        senkou_a = self.senkou_span_a[-1]
        senkou_b = self.senkou_span_b[-1]

        # Conditions for a long trade
        if (price > senkou_a and 
            price > senkou_b and
            crossover(self.tenkan_sen, self.kijun_sen)):
            self.buy()

        # Conditions for closing a long trade
        elif self.position.is_long and (
              price < kijun or 
              price < senkou_b or
              crossover(self.kijun_sen, self.tenkan_sen)):
            self.position.close()

        # Conditions for a short trade
        if (price < senkou_a and 
            price < senkou_b and
            crossover(self.kijun_sen, self.tenkan_sen)):
            self.sell()

        # Conditions for closing a short trade
        elif self.position.is_short and (
              price > kijun or 
              price > senkou_a or
              crossover(self.tenkan_sen, self.kijun_sen)):
            self.position.close()
