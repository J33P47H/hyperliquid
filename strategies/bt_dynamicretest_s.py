import pandas as pd
import numpy as np
from backtesting import Strategy

class DynamicRetestStrategy(Strategy):
    """
    A strategy that identifies price zones and trades retests of these zones.
    Uses pure price action and zone retests with dynamic risk management.
    """
    
    # Define parameters as class variables first
    consolidation_span = 3  # Number of bars for zone determination
    
    # Parameter configuration for optimization
    param_config = {
        "consolidation_span": {
            "default": 3,
            "range": list(range(2, 6))  # 2 to 5 bars
        }
    }

    def init(self):
        # Calculate zone boundaries using rolling windows
        def calculate_zones(high, low, span):
            # Convert inputs to pandas Series
            high = pd.Series(high)
            low = pd.Series(low)
            
            # Calculate rolling max/min with minimum periods
            zone_top = high.rolling(window=span, min_periods=1).max()
            zone_bottom = low.rolling(window=span, min_periods=1).min()
            
            # Ensure zones are valid
            zone_top = zone_top.replace([np.inf, -np.inf], np.nan)
            zone_bottom = zone_bottom.replace([np.inf, -np.inf], np.nan)
            
            # Fill any NaN values with the current high/low
            zone_top = zone_top.fillna(high)
            zone_bottom = zone_bottom.fillna(low)
            
            return zone_top, zone_bottom

        # Calculate indicators using the wrapper
        self.zone_top, self.zone_bottom = self.I(lambda: calculate_zones(
            self.data.High,
            self.data.Low,
            self.consolidation_span
        ), name='zones')

        # Optional SMA for trend context
        self.sma20 = self.I(lambda: pd.Series(self.data.Close).ewm(span=20, min_periods=1, adjust=False).mean(), name='sma20')

    def next(self):
        if len(self.data) < max(self.consolidation_span, 20):
            return

        # Get current prices and zone levels
        price = self.data.Close[-1]
        curr_zone_top = self.zone_top[-1]
        curr_zone_bottom = self.zone_bottom[-1]

        # Validate all prices and levels
        if not (0 < price < float('inf') and 
                0 < curr_zone_top < float('inf') and 
                0 < curr_zone_bottom < float('inf') and
                curr_zone_bottom <= curr_zone_top):
            return

        # Simple trend identification using last 3 bars
        trend = 'none'
        if len(self.data) >= 3:
            if (self.data.Close[-1] > self.data.Close[-2] > self.data.Close[-3] and
                price > self.sma20[-1]):
                trend = 'up'
            elif (self.data.Close[-1] < self.data.Close[-2] < self.data.Close[-3] and
                  price < self.sma20[-1]):
                trend = 'down'

        # Only consider new trades if not in a position
        if not self.position:
            # Long setup in uptrend
            if trend == 'up' and self.data.Close[-1] > self.data.Open[-1]:
                if curr_zone_bottom <= price <= curr_zone_top:
                    self.buy()

            # Short setup in downtrend
            elif trend == 'down' and self.data.Close[-1] < self.data.Open[-1]:
                if curr_zone_bottom <= price <= curr_zone_top:
                    self.sell()

        # Exit Logic - using trend reversal
        elif self.position:
            if self.position.is_long and trend == 'down':
                self.position.close()
            elif self.position.is_short and trend == 'up':
                self.position.close() 