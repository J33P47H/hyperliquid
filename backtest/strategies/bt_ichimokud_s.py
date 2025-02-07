import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover

class IchimokudStrategy(Strategy):
    # Class-level parameter configuration with start/end/step structure
    param_config = {
        "tenkan_period": {
            "default": 9,
            "start": 7,
            "end": 20,
            "step": 2
        },
        "kijun_period": {
            "default": 26,
            "start": 20,
            "end": 40,
            "step": 4
        },
        "senkou_span_b_period": {
            "default": 52,
            "start": 40,
            "end": 75,
            "step": 5
        },
        "displacement": {
            "default": 26,
            "start": 20,
            "end": 35,
            "step": 3
        }
    }
    
    # Default parameter values
    tenkan_period = param_config["tenkan_period"]["default"]
    kijun_period = param_config["kijun_period"]["default"]
    senkou_span_b_period = param_config["senkou_span_b_period"]["default"]
    displacement = param_config["displacement"]["default"]

    def init(self):
        # Set instance parameters from class attributes or defaults
        self.tenkan_period = getattr(self, 'tenkan_period', 9)
        self.kijun_period = getattr(self, 'kijun_period', 26)
        self.senkou_span_b_period = getattr(self, 'senkou_span_b_period', 52)
        self.displacement = getattr(self, 'displacement', 26)

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
        
        # Check cloud-based exit conditions if we have a position
        if self.position.is_long:
            # Exit long position if price enters the cloud from above
            if price <= cloud_upper:
                self.position.close()
                return
                
        elif self.position.is_short:
            # Exit short position if price enters the cloud from below
            if price >= cloud_lower:
                self.position.close()
                return
                    
        # Entry conditions (only if we are not in a position)
        if not self.position:
            # Long Entry Conditions:
            # 1. Price must be above both Senkou spans (above the cloud)
            # 2. Tenkan/Kijun bullish crossover
            # 3. Chikou span confirms trend (price N periods ago below current Chikou)
            if (price > self.senkou_span_a[-1] and 
                price > self.senkou_span_b[-1] and
                crossover(self.tenkan_sen, self.kijun_sen) and
                len(self.data) > self.displacement and
                self.data.Close[-self.displacement] < self.chikou_span[-1]):
                
                self.buy()
                
            # Short Entry Conditions:
            # 1. Price must be below both Senkou spans (below the cloud)
            # 2. Kijun/Tenkan bearish crossover
            # 3. Chikou span confirms trend (price N periods ago above current Chikou)
            elif (price < self.senkou_span_a[-1] and 
                  price < self.senkou_span_b[-1] and
                  crossover(self.kijun_sen, self.tenkan_sen) and
                  len(self.data) > self.displacement and
                  self.data.Close[-self.displacement] > self.chikou_span[-1]):
                
                self.sell() 