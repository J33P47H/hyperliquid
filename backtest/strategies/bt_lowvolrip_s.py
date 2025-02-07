import numpy as np
import pandas as pd
from backtesting import Strategy

def calculate_log_atr(high, low, close, period=14):
    ln_high = np.log(high)
    ln_low = np.log(low)
    ln_close = np.log(close)
    prev_close = pd.Series(ln_close).shift(1)

    tr1 = ln_high - ln_low
    tr2 = (ln_high - prev_close).abs()
    tr3 = (ln_low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)
    return tr.ewm(span=period, min_periods=1, adjust=False).mean()

def low_vol_rip_signal(close, vol_series, lookback=100, q_low=0.05):
    sig = np.zeros(len(close))
    trap_active = False
    trap_low_idx = -1
    trap_high_idx = -1

    vol_q_low = vol_series.rolling(lookback).quantile(q_low)

    for i in range(1, len(sig)):
        if not trap_active:
            if vol_series[i] < vol_q_low[i]:
                trap_active = True
                trap_low_idx = i
                trap_high_idx = i
        else:
            if vol_series[i] >= vol_q_low[i]:
                trap_active = False
                sig[i] = 0
                continue

            # Track min/max close in the “trap” zone
            if close[i] < close[trap_low_idx]:
                trap_low_idx = i
            if close[i] > close[trap_high_idx]:
                trap_high_idx = i

            # If we break above the trap’s max → go long
            if close[i] > close[trap_high_idx]:
                sig[i] = 1
                trap_active = False

            # If we break below the trap’s min → go short
            elif close[i] < close[trap_low_idx]:
                sig[i] = -1
                trap_active = False

    return pd.Series(sig, index=close.index)

class LowVolRipStrategy(Strategy):
    log_atr_period = 14
    vol_lookback   = 100
    quantile_level = 0.05

    param_config = {
        "log_atr_period": {
            "default": 14,
            "range": [10, 14, 20, 30]
        },
        "vol_lookback": {
            "default": 100,
            "range": [50, 100, 150, 200]
        },
        "quantile_level": {
            "default": 0.05,
            "range": [0.01, 0.05, 0.1]
        }
    }

    def init(self):
        high = self.data.High
        low  = self.data.Low
        close = self.data.Close

        def _log_atr():
            return calculate_log_atr(high, low, close, self.log_atr_period)

        self.log_atr = self.I(_log_atr, name='log_atr')

        def _vol():
            return (np.log(high) - np.log(low)) / self.log_atr

        self.vol_measure = self.I(_vol, name='low_vol_measure')

        def _signal():
            c = pd.Series(close, index=self.data.index)
            v = pd.Series(self.vol_measure, index=self.data.index)
            return low_vol_rip_signal(
                c, v, 
                lookback=self.vol_lookback, 
                q_low=self.quantile_level
            )

        self.signal = self.I(_signal, name='rip_signal')

    def next(self):
        sig = self.signal[-1]

        if sig == 0 and self.position:
            self.position.close()

        elif sig == 1:
            if self.position.is_short:
                self.position.close()
            if not self.position:
                self.buy()

        elif sig == -1:
            if self.position.is_long:
                self.position.close()
            if not self.position:
                self.sell()
