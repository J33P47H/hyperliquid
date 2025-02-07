import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover
import talib

# ---------------------------------------------------------------------
# Indicator helper functions
# ---------------------------------------------------------------------
def rsi(series, period=14):
    """
    Compute RSI indicator.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    # To avoid division by zero, add a very small epsilon
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def true_range(high, low, close):
    """
    True Range = max of:
        - currentHigh - currentLow
        - abs(currentHigh - prevClose)
        - abs(currentLow - prevClose)
    """
    shifted_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - shifted_close).abs()
    tr3 = (low - shifted_close).abs()
    return pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

def average_true_range(df, period=14):
    """
    Standard ATR calculation on a DataFrame with columns: High, Low, Close.
    """
    tr = true_range(df['High'], df['Low'], df['Close'])
    return tr.rolling(period).mean()

def adx(df, period=14):
    """
    Approximate ADX calculation.
    Requires df to have columns: High, Low, Close.
    Returns a pd.Series of ADX values.
    """
    # Calculate TR, +DM, -DM
    high, low, close = df['High'], df['Low'], df['Close']
    shift_high, shift_low = high.shift(1), low.shift(1)
    
    plus_dm = (high - shift_high).clip(lower=0)
    minus_dm = (shift_low - low).clip(lower=0)

    # Smooth them
    atr_series = average_true_range(df, period)
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr_series + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr_series + 1e-10))

    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10) ) * 100
    adx_val = dx.rolling(period).mean()  # smooth the DX
    return adx_val

def rolling_mean(series, period=20):
    return series.rolling(period).mean()

# ---------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------
class QuickpStrategy(Strategy):
    """
    A quick-profit RSI + ATR strategy in the style of the Bollinger example.
    It includes parameters for RSI, ATR-based exit, volume filter, ADX filter, etc.
    Uses ATR for position sizing and risk management.
    """
    # -- Strategy parameters as class variables --
    n1 = 20                # Not used here, but kept for compatibility
    n2 = 50                # Not used here, but kept for compatibility
    n_std_dev = 2          # Not used here, but kept for compatibility

    atr_period = 14
    atr_exit_mult = 1.5
    atr_threshold = 0.0

    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30

    adx_period = 14
    adx_max = 25

    min_volume_ratio = 1.0
    profit_target_rr = 2.0

    # -- Param configuration for your system --
    param_config = {
        "n1": {"default": 20, "range": [10, 20, 30, 40]},
        "n2": {"default": 50, "range": [30, 50, 70, 100]},
        "n_std_dev": {"default": 2, "range": [1.5, 2, 2.5, 3]},

        "atr_period": {"default": 14, "range": [10, 14, 20]},
        "atr_exit_mult": {"default": 1.5, "range": [1.0, 1.5, 2.0, 2.5]},
        "atr_threshold": {"default": 0.0, "range": [0.0, 0.1, 0.2, 0.3]},

        "rsi_period": {"default": 14, "range": [10, 14, 20]},
        "rsi_overbought": {"default": 70, "range": [60, 70, 80]},
        "rsi_oversold": {"default": 30, "range": [20, 30, 40]},

        "adx_period": {"default": 14, "range": [10, 14, 20]},
        "adx_max": {"default": 25, "range": [20, 25, 30]},

        "min_volume_ratio": {"default": 1.0, "range": [0.8, 1.0, 1.2]},
        "profit_target_rr": {"default": 2.0, "range": [1.5, 2.0, 2.5, 3.0]}
    }

    def init(self):
        """
        Define indicators once per bar. 
        Use self.I(...) so backtesting.py can handle them properly.
        """
        # (Optional) Re-assign for clarity (in case they're changed by optimization)
        self.atr_period = getattr(self, 'atr_period', 14)
        self.atr_exit_mult = getattr(self, 'atr_exit_mult', 1.5)
        self.atr_threshold = getattr(self, 'atr_threshold', 0.0)

        self.rsi_period = getattr(self, 'rsi_period', 14)
        self.rsi_overbought = getattr(self, 'rsi_overbought', 70)
        self.rsi_oversold = getattr(self, 'rsi_oversold', 30)

        self.adx_period = getattr(self, 'adx_period', 14)
        self.adx_max = getattr(self, 'adx_max', 25)
        self.min_volume_ratio = getattr(self, 'min_volume_ratio', 1.0)
        self.profit_target_rr = getattr(self, 'profit_target_rr', 2.0)

        # Calculate indicators using TA-Lib
        # 1) RSI
        self.rsi_series = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        
        # 2) ATR
        self.atr_series = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                                timeperiod=self.atr_period)
        
        # 3) ADX
        self.adx_series = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, 
                                timeperiod=self.adx_period)

        # 4) Rolling average volume
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)

        # Initialize tracking variables for position management
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0

    def next(self):
        # 1) Basic filters: skip if ATR is too low, ADX too high, or volume too low
        curr_atr = self.atr_series[-1]
        curr_adx = self.adx_series[-1]
        curr_vol = self.data.Volume[-1]
        curr_price = self.data.Close[-1]
        vol_filter = (curr_vol >= self.vol_ma[-1] * self.min_volume_ratio)

        # Skip if any indicator is NaN (not enough data yet)
        if pd.isna(curr_atr) or pd.isna(curr_adx) or pd.isna(self.rsi_series[-1]):
            return

        # If ATR is below threshold, skip
        if curr_atr < self.atr_threshold:
            return

        # If ADX is above our "max," skip
        if curr_adx > self.adx_max:
            return

        # If volume is too low, skip
        if not vol_filter:
            return

        # 2) Calculate RSI-based signals
        curr_rsi = self.rsi_series[-1]

        # Check existing position for exits
        if self.position:
            # Check stop loss and take profit
            if self.position.is_long:
                if curr_price <= self.stop_loss or curr_price >= self.take_profit:
                    self.position.close()
            elif self.position.is_short:
                if curr_price >= self.stop_loss or curr_price <= self.take_profit:
                    self.position.close()
            return

        # If flat, consider new entries
        if not self.position:
            # Risk in terms of ATR
            risk = curr_atr * self.atr_exit_mult

            # LONG signal: RSI < oversold
            if curr_rsi < self.rsi_oversold:
                self.entry_price = curr_price
                self.stop_loss = curr_price - risk
                self.take_profit = curr_price + (risk * self.profit_target_rr)
                self.buy()

            # SHORT signal: RSI > overbought
            elif curr_rsi > self.rsi_overbought:
                self.entry_price = curr_price
                self.stop_loss = curr_price + risk
                self.take_profit = curr_price - (risk * self.profit_target_rr)
                self.sell()
