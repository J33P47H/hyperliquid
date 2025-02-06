import numpy as np
import pandas as pd

from backtesting import Strategy

def calculate_atr(high, low, close, period=14):
    """
    Calculate ATR (average true range) via exponential smoothing.
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    close_prev = close.shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Replace infinities, fill NaNs
    tr = tr.replace([np.inf, -np.inf], np.nan).fillna(0)
    atr = tr.ewm(span=period, min_periods=1, adjust=False).mean()
    return atr

def calculate_norm_range(high, low, close, atr, period=336):
    """
    Replicates the 'norm_range' from the video:
    norm_range = (ln(high) - ln(low)) / ATR_of_ln_prices

    For simplicity, we use the ATR of ln(close) as in the original script, 
    but you can adjust to suit your data if needed.
    """
    # Convert to logs
    ln_high = np.log(high)
    ln_low = np.log(low)
    # 'atr' is already computed on log prices for consistency
    norm_range = (ln_high - ln_low) / atr
    return norm_range

def hawkes_process(series: pd.Series, kappa: float):
    """
    Hawkes-like decay process from the video.
    """
    assert kappa > 0.0, "kappa must be > 0"
    alpha = np.exp(-kappa)
    arr = series.to_numpy()

    output = np.zeros(len(series))
    output[:] = np.nan

    for i in range(1, len(series)):
        if np.isnan(output[i - 1]):
            output[i] = arr[i]
        else:
            output[i] = output[i - 1] * alpha + arr[i]

    return pd.Series(output, index=series.index) * kappa

def vol_signal(close: pd.Series, vol_hawkes: pd.Series, lookback: int):
    """
    Returns a signal array from -1, 0, or 1, as in the video. 
    - If Hawkes-based 'vol' < 5th percentile → reset to 'flat' (0).
    - If Hawkes-based 'vol' crosses > 95th percentile, go long (1) 
      or short (-1) depending on price change since last sub-5% reading.
    """
    signal = np.zeros(len(close))

    q05 = vol_hawkes.rolling(lookback).quantile(0.05)
    q95 = vol_hawkes.rolling(lookback).quantile(0.95)

    last_below = -1
    curr_sig = 0

    for i in range(len(signal)):
        # If current hawkes vol is below 5th percentile, set current sig = 0, record index
        if vol_hawkes.iloc[i] < q05.iloc[i]:
            last_below = i
            curr_sig = 0

        # If we jump above the 95th percentile AND we were not above it the previous candle
        # AND we have a valid last_below
        if (vol_hawkes.iloc[i] > q95.iloc[i] and
            vol_hawkes.iloc[i - 1] <= q95.iloc[i - 1] and
            last_below > 0):

            change = close.iloc[i] - close.iloc[last_below]
            # If net price change since last sub-5% reading is positive, go long
            if change > 0.0:
                curr_sig = 1
            else:
                curr_sig = -1

        signal[i] = curr_sig

    return pd.Series(signal, index=close.index)

class HawkesProStrategy(Strategy):
    """
    A strategy that detects volatility spikes via a 'Hawkes-like' 
    exponential decay process on normalized ranges. 
    It flips long/short based on signals from vol_signal().
    """

    # ----------- STRATEGY PARAMETERS -----------
    kappa = 0.1           # Decay factor in Hawkes process
    norm_lookback = 336   # Period for ATR on log prices
    signal_lookback = 168 # Rolling period for 5%/95% thresholds

    # We can expose these in a param_config dict
    param_config = {
        "kappa": {
            "default": 0.1,
            "range": [0.01, 0.05, 0.1, 0.25, 0.5]
        },
        "norm_lookback": {
            "default": 336,
            "range": [24, 48, 96, 168, 336]
        },
        "signal_lookback": {
            "default": 168,
            "range": [24, 48, 96, 168, 336]
        }
    }

    def init(self):
        """
        Register our indicators with backtesting engine.
        Note: backtesting.py expects arrays or callables returning arrays.
        We'll compute:
          - atr_log (on log prices),
          - norm_range,
          - hawkes_vol,
          - signal
        """
        high = self.data.High
        low  = self.data.Low
        close = self.data.Close

        # 1) ATR on log prices
        def _atr_log():
            ln_high = np.log(high)
            ln_low = np.log(low)
            ln_close = np.log(close)
            return calculate_atr(ln_high, ln_low, ln_close, period=self.norm_lookback)

        self.atr_log = self.I(_atr_log, name='atr_log')

        # 2) Normalized range
        def _norm_range():
            return calculate_norm_range(high, low, close, self.atr_log, self.norm_lookback)
        self.norm_range = self.I(_norm_range, name='norm_range')

        # 3) Hawkes process on norm_range
        def _hawkes():
            series = pd.Series(self.norm_range, index=self.data.index)
            return hawkes_process(series, self.kappa)
        self.hawkes_vol = self.I(_hawkes, name='hawkes_vol')

        # 4) The final signal array
        def _signal():
            # vol_signal expects pandas Series
            close_s = pd.Series(close, index=self.data.index)
            hawk_s  = pd.Series(self.hawkes_vol, index=self.data.index)
            return vol_signal(close_s, hawk_s, self.signal_lookback)
        self.signal = self.I(_signal, name='signal')

    def next(self):
        """
        For each bar:
         - If signal == 1, go long
         - If signal == -1, go short
         - If signal == 0, close position
        We skip advanced stops/targets here, but you can incorporate them if desired.
        """
        sig = self.signal[-1]  # current signal

        # Flatten if the signal is 0
        if sig == 0:
            if self.position:
                self.position.close()
            return

        # Go long if the current signal is 1
        if sig == 1:
            # If we already have a position but it's short, close it first
            if self.position.is_short:
                self.position.close()
            # If not in any position, buy
            if not self.position:
                self.buy()

        # Go short if the current signal is -1
        elif sig == -1:
            # If we already have a position but it's long, close it first
            if self.position.is_long:
                self.position.close()
            # If not in any position, sell
            if not self.position:
                self.sell()
