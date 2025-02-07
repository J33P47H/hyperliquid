import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover

class IchimokusStrategy(Strategy):
    # Class-level parameter configuration, including trailing stop percentage
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
        },
        "trailing_stop_pct": {
            "default": 0.02,  # 2% trailing stop by default
            "range": [0.01, 0.02, 0.03]
        }
    }
    
    # Default parameter values
    tenkan_period = param_config["tenkan_period"]["default"]
    kijun_period = param_config["kijun_period"]["default"]
    senkou_span_b_period = param_config["senkou_span_b_period"]["default"]
    displacement = param_config["displacement"]["default"]
    trailing_stop_pct = param_config["trailing_stop_pct"]["default"]

    def init(self):
        # Set instance parameters from class attributes or defaults
        self.tenkan_period = getattr(self, 'tenkan_period', 9)
        self.kijun_period = getattr(self, 'kijun_period', 26)
        self.senkou_span_b_period = getattr(self, 'senkou_span_b_period', 52)
        self.displacement = getattr(self, 'displacement', 26)
        self.trailing_stop_pct = getattr(self, 'trailing_stop_pct', 0.02)

        # Initialize trailing stop variables
        self.highest_price = 0
        self.lowest_price = float('inf')

        # Helper function to compute the Ichimoku line (midpoint of rolling high/low)
        def calculate_ichimoku_line(high, low, period):
            high_values = pd.Series(high).rolling(period).max()
            low_values = pd.Series(low).rolling(period).min()
            return (high_values + low_values) / 2

        # Calculate main Ichimoku components
        self.tenkan_sen = self.I(calculate_ichimoku_line, self.data.High, self.data.Low, self.tenkan_period)
        self.kijun_sen = self.I(calculate_ichimoku_line, self.data.High, self.data.Low, self.kijun_period)
        
        # Calculate Senkou Span A and shift it forward (displacement)
        senkou_span_a = (self.tenkan_sen + self.kijun_sen) / 2
        self.senkou_span_a = self.I(lambda x: np.roll(x, self.displacement), senkou_span_a)
        
        # Calculate Senkou Span B and shift it forward (displacement)
        senkou_span_b = self.I(calculate_ichimoku_line, self.data.High, self.data.Low, self.senkou_span_b_period)
        self.senkou_span_b = self.I(lambda x: np.roll(x, self.displacement), senkou_span_b)
        
        # Calculate Chikou Span (lagging span)
        self.chikou_span = self.I(lambda x: np.roll(x, -self.displacement), self.data.Close)
        
    def next(self):
        # Ensure we have enough data before processing signals
        required_bars = max(self.displacement, self.senkou_span_b_period)
        if len(self.data) < required_bars:
            return

        price = self.data.Close[-1]
        
        # Define cloud boundaries using current values
        cloud_upper = max(self.senkou_span_a[-1], self.senkou_span_b[-1])
        cloud_lower = min(self.senkou_span_a[-1], self.senkou_span_b[-1])
        
        # Check trailing stop conditions if we have a position
        if self.position.is_long:
            # Update highest seen price
            self.highest_price = max(self.highest_price, price)
            # Calculate trailing stop price
            trailing_stop = self.highest_price * (1 - self.trailing_stop_pct)
            
            # Close position if price drops below trailing stop or cloud exit condition
            if price < trailing_stop or price < cloud_upper:
                self.position.close()
                self.highest_price = 0  # Reset tracking
                return
                
        elif self.position.is_short:
            # Update lowest seen price
            self.lowest_price = min(self.lowest_price, price)
            # Calculate trailing stop price
            trailing_stop = self.lowest_price * (1 + self.trailing_stop_pct)
            
            # Close position if price rises above trailing stop or cloud exit condition
            if price > trailing_stop or price > cloud_lower:
                self.position.close()
                self.lowest_price = float('inf')  # Reset tracking
                return
                    
        # Entry conditions (only if we are not in a position)
        if not self.position:
            # Long Entry: Price must be above both Senkou spans, and we require a Tenkan/Kijun bullish crossover.
            if (price > self.senkou_span_a[-1] and 
                price > self.senkou_span_b[-1] and
                crossover(self.tenkan_sen, self.kijun_sen) and
                len(self.data) > self.displacement and
                self.data.Close[-self.displacement] < self.chikou_span[-1]):
                
                self.highest_price = price  # Initialize trailing stop tracking
                self.buy()
                
            # Short Entry: Price must be below both Senkou spans, and we require a Kijun/Tenkan bearish crossover.
            elif (price < self.senkou_span_a[-1] and 
                  price < self.senkou_span_b[-1] and
                  crossover(self.kijun_sen, self.tenkan_sen) and
                  len(self.data) > self.displacement and
                  self.data.Close[-self.displacement] > self.chikou_span[-1]):
                
                self.lowest_price = price  # Initialize trailing stop tracking
                self.sell()
