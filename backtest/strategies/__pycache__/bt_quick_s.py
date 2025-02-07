# quick_profit_strategy.py
import numpy as np
import pandas as pd

# pip install backtesting
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG  # Just for sample data

from backtesting.lib import SignalStrategy, TrailingStrategy
from backtesting.lib import resample_apply

# If you have real 1m crypto data in a DataFrame, ensure it has:
#   - 'Open', 'High', 'Low', 'Close', 'Volume' columns
#   - A DateTimeIndex
# Example:
# df = pd.read_csv('your_1m_data.csv', parse_dates=True, index_col='Date')

def rsi(series, period=14):
    """Compute RSI indicator."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

class QuickStrategy(Strategy):
    """
    1) Buys when RSI < 30 (oversold).
    2) Sells (shorts) when RSI > 70 (overbought).
    3) Uses a fixed stop loss and take profit.
    4) Uses a trailing stop to lock in profits.
    """
    # Strategy parameters (can be optimized)
    rsi_period = 14
    rsi_lower = 30
    rsi_upper = 70
    stop_loss_pct = 0.5  # 0.5% stop loss
    take_profit_pct = 1.0  # 1.0% take profit
    trailing_stop_pct = 0.8  # 0.8% trailing stop

    def init(self):
        # Compute RSI once per bar
        self.rsi_series = self.I(rsi, self.data.Close, self.rsi_period)

    def next(self):
        price = self.data.Close[-1]
        current_position = self.position

        # Check if we are flat:
        if not current_position:
            # Go long if RSI < rsi_lower
            if self.rsi_series[-1] < self.rsi_lower:
                # Use bracket orders with stop loss, take profit, and trailing stop
                sl_price = price * (1 - self.stop_loss_pct / 100)
                tp_price = price * (1 + self.take_profit_pct / 100)
                self.buy(
                    sl=sl_price,
                    tp=tp_price,
                    trail_amount=price * self.trailing_stop_pct / 100  # trailing stop
                )

            # Go short if RSI > rsi_upper
            elif self.rsi_series[-1] > self.rsi_upper:
                sl_price = price * (1 + self.stop_loss_pct / 100)
                tp_price = price * (1 - self.take_profit_pct / 100)
                self.sell(
                    sl=sl_price,
                    tp=tp_price,
                    trail_amount=price * self.trailing_stop_pct / 100
                )
        else:
            # If you want partial exit logic or manual exit signals, handle it here
            pass