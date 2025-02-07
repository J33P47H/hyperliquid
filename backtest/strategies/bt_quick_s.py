import pandas as pd
import numpy as np
from backtesting import Strategy

def rsi_func(series, period=14):
    """
    Compute RSI indicator.
    """
    s = pd.Series(series)  # Ensure input is a pandas Series
    delta = s.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()
    rs = gain / (loss + 1e-10)  # add small epsilon to avoid division by zero
    return 100 - (100 / (1 + rs))

class QuickStrategy(Strategy):
    """
    1) Buys when RSI < 30 (oversold).
    2) Sells (shorts) when RSI > 70 (overbought).
    3) Uses a fixed stop loss and take profit.
    4) Uses a trailing stop to lock in profits.
    """
    # --- Strategy Parameters ---
    rsi_period = 14
    rsi_lower = 30
    rsi_upper = 70
    stop_loss_pct = 0.5
    take_profit_pct = 1.0
    trailing_stop_pct = 0.8

    # --- Param Configuration ---
    param_config = {
        "rsi_period":        {"default": 14, "range": [10, 14, 20]},
        "rsi_lower":         {"default": 30, "range": [20, 30, 40]},
        "rsi_upper":         {"default": 70, "range": [60, 70, 80]},
        "stop_loss_pct":     {"default": 0.5, "range": [0.3, 0.5, 1.0]},
        "take_profit_pct":   {"default": 1.0, "range": [0.5, 1.0, 2.0]},
        "trailing_stop_pct": {"default": 0.8, "range": [0.3, 0.5, 0.8, 1.0]},
    }

    def init(self):
        # Re-bind parameters in case they're overridden by optimization
        self.rsi_period        = getattr(self, 'rsi_period', 14)
        self.rsi_lower         = getattr(self, 'rsi_lower', 30)
        self.rsi_upper         = getattr(self, 'rsi_upper', 70)
        self.stop_loss_pct     = getattr(self, 'stop_loss_pct', 0.5)
        self.take_profit_pct   = getattr(self, 'take_profit_pct', 1.0)
        self.trailing_stop_pct = getattr(self, 'trailing_stop_pct', 0.8)
        
        # Compute RSI once per bar using our rsi_func
        self.rsi_series = self.I(rsi_func, self.data.Close, self.rsi_period)
        
        # Initialize variables for trailing stop
        self.entry_price = 0
        self.highest_price = 0
        self.lowest_price = float('inf')

    def next(self):
        price = self.data.Close[-1]
        current_position = self.position

        # Check trailing stop conditions if we have a position
        if current_position.is_long:
            # Update highest seen price
            self.highest_price = max(self.highest_price, price)
            # Calculate trailing stop price
            trailing_stop = self.highest_price * (1 - self.trailing_stop_pct / 100)
            
            # Close position if price drops below trailing stop
            if price < trailing_stop:
                self.position.close()
                self.highest_price = 0  # Reset tracking
                
        elif current_position.is_short:
            # Update lowest seen price
            self.lowest_price = min(self.lowest_price, price)
            # Calculate trailing stop price
            trailing_stop = self.lowest_price * (1 + self.trailing_stop_pct / 100)
            
            # Close position if price rises above trailing stop
            if price > trailing_stop:
                self.position.close()
                self.lowest_price = float('inf')  # Reset tracking

        # Check if we are flat (no position)
        if not current_position:
            # Go long if RSI < rsi_lower
            if self.rsi_series[-1] < self.rsi_lower:
                self.entry_price = price
                self.highest_price = price
                sl_price = price * (1 - self.stop_loss_pct / 100)
                tp_price = price * (1 + self.take_profit_pct / 100)
                self.buy(sl=sl_price, tp=tp_price)

            # Go short if RSI > rsi_upper
            elif self.rsi_series[-1] > self.rsi_upper:
                self.entry_price = price
                self.lowest_price = price
                sl_price = price * (1 + self.stop_loss_pct / 100)
                tp_price = price * (1 - self.take_profit_pct / 100)
                self.sell(sl=sl_price, tp=tp_price)
