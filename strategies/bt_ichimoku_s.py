import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover

class IchimokuStrategy(Strategy):
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
        self.tenkan_sen = self.I(lambda x: calculate_ichimoku_line(x.High, x.Low, self.tenkan_period), self.data)
        self.kijun_sen = self.I(lambda x: calculate_ichimoku_line(x.High, x.Low, self.kijun_period), self.data)
        
        # Calculate Senkou Spans
        senkou_span_a = (self.tenkan_sen + self.kijun_sen) / 2
        self.senkou_span_a = self.I(lambda x: np.roll(senkou_span_a, self.displacement), self.data)
        
        senkou_span_b = self.I(lambda x: calculate_ichimoku_line(x.High, x.Low, self.senkou_span_b_period), self.data)
        self.senkou_span_b = self.I(lambda x: np.roll(senkou_span_b, self.displacement), self.data)
        
        # Calculate Chikou Span
        self.chikou_span = self.I(lambda x: np.roll(x.Close, -self.displacement), self.data)
        
    def next(self):
        # Wait for enough data
        required_bars = max(self.displacement, self.senkou_span_b_period)
        if len(self.data) < required_bars:
            return
            
        # Conditions for a long trade
        if (self.data.Close[-1] > self.senkou_span_a[-1] and 
            self.data.Close[-1] > self.senkou_span_b[-1] and
            crossover(self.tenkan_sen, self.kijun_sen) and
            len(self.data) > self.displacement and
            self.data.Close[-self.displacement] < self.chikou_span[-1]):
            self.buy()
            
        # Conditions for closing a long trade
        elif (self.position.is_long and 
              ((self.data.Close[-1] < self.kijun_sen[-1]) or 
               (self.data.Close[-1] < self.senkou_span_b[-1]) or
               crossover(self.kijun_sen, self.tenkan_sen))):
            self.position.close()
        
        # Conditions for a short trade
        elif (self.data.Close[-1] < self.senkou_span_a[-1] and 
              self.data.Close[-1] < self.senkou_span_b[-1] and
              crossover(self.kijun_sen, self.tenkan_sen) and
              len(self.data) > self.displacement and
              self.data.Close[-self.displacement] > self.chikou_span[-1]):
            self.sell()
            
        # Conditions for closing a short trade
        elif (self.position.is_short and 
              ((self.data.Close[-1] > self.kijun_sen[-1]) or 
               (self.data.Close[-1] > self.senkou_span_a[-1]) or
               crossover(self.tenkan_sen, self.kijun_sen))):
            self.position.close() 