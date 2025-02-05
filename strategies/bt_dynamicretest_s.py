import pandas as pd
import numpy as np
from backtesting import Strategy

class DynamicRetestStrategy(Strategy):
    """
    A strategy that identifies price zones and trades retests of these zones.
    Uses pure price action and zone retests with dynamic risk management.
    """
    
    # Define parameters as class variables first
    risk_reward = 25      # Effective risk/reward ratio = risk_reward/10 (default 25 -> 2.5:1)
    risk_percent = 1      # Effective risk per trade = risk_percent/100 (default 1 -> 1%)
    consolidation_span = 3  # Number of bars for zone determination
    
    # Parameter configuration for optimization
    param_config = {
        "risk_reward": {
            "default": 25,
            "range": list(range(25, 31))  # 2.5:1 to 3.0:1
        },
        "risk_percent": {
            "default": 1,
            "range": [1, 2]  # 1% to 2%
        },
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

        # Convert parameters to effective values
        eff_risk_reward = self.risk_reward / 10.0  # e.g., 25 becomes 2.5:1
        eff_risk_percent = self.risk_percent / 100.0  # e.g., 1 becomes 1%

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
                    stop_loss = curr_zone_bottom * 0.99  # 1% below zone bottom
                    risk_per_unit = price - stop_loss
                    
                    if risk_per_unit > 0:
                        risk_amount = self.equity * eff_risk_percent
                        position_size = int(risk_amount / risk_per_unit)
                        if position_size > 0:
                            target = price + (risk_per_unit * eff_risk_reward)
                            # Validate target price
                            if 0 < target < float('inf'):
                                self.buy(size=position_size, sl=stop_loss, tp=target)

            # Short setup in downtrend
            elif trend == 'down' and self.data.Close[-1] < self.data.Open[-1]:
                if curr_zone_bottom <= price <= curr_zone_top:
                    stop_loss = curr_zone_top * 1.01  # 1% above zone top
                    risk_per_unit = stop_loss - price
                    
                    if risk_per_unit > 0:
                        risk_amount = self.equity * eff_risk_percent
                        position_size = int(risk_amount / risk_per_unit)
                        if position_size > 0:
                            target = price - (risk_per_unit * eff_risk_reward)
                            # Validate target price
                            if 0 < target < float('inf'):
                                self.sell(size=position_size, sl=stop_loss, tp=target)

        # Exit Logic - using stop loss and take profit only
        # No additional exit conditions needed as we're using strict SL/TP 