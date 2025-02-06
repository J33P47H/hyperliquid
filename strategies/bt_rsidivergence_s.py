import numpy as np
from backtesting import Strategy

# --- Custom Indicator Implementations ---

def custom_RSI(prices, period=14):
    """
    Calculate RSI using Wilder's smoothing method.
    Returns an array of RSI values.
    """
    prices = np.array(prices, dtype=float)
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    rsi = np.empty_like(prices)
    avg_gain = np.empty_like(prices)
    avg_loss = np.empty_like(prices)
    
    # Set the first period-1 values to NaN
    rsi[:period-1] = np.nan
    avg_gain[:period-1] = np.nan
    avg_loss[:period-1] = np.nan
    
    # First average gain and loss: simple mean over first period
    avg_gain[period-1] = np.mean(gain[1:period+1])
    avg_loss[period-1] = np.mean(loss[1:period+1])
    rsi[period-1] = 100 - (100 / (1 + avg_gain[period-1] / (avg_loss[period-1] + 1e-10)))
    
    for i in range(period, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
        rs = avg_gain[i] / (avg_loss[i] + 1e-10)
        rsi[i] = 100 - (100 / (1 + rs))
        
    return rsi

def custom_ATR(high, low, close, period=14):
    """
    Calculate the Average True Range (ATR).
    Returns an array of ATR values.
    """
    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)
    close = np.array(close, dtype=float)
    
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    
    atr = np.empty_like(tr)
    atr[:period-1] = np.nan
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def find_divergences(price_data, rsi_data, window=5):
    """
    Look for bullish and bearish divergences.
    Returns two arrays (bullish_div, bearish_div) of the same length as the input data.
    """
    bullish_div = []
    bearish_div = []
    
    price_arr = np.array(price_data, dtype=float)
    rsi_arr = np.array(rsi_data, dtype=float)
    
    for i in range(window, len(price_arr)):
        price_window = price_arr[i-window:i+1]
        rsi_window = rsi_arr[i-window:i+1]
        
        # If there are any NaNs, skip the divergence check
        if np.isnan(price_window).any() or np.isnan(rsi_window).any():
            bullish_div.append(0)
            bearish_div.append(0)
            continue
        
        # Bullish divergence: new low in price with a higher low in RSI.
        if price_window[-1] <= price_window.min() and rsi_window[-1] > rsi_window.min():
            bullish_div.append(1)
        else:
            bullish_div.append(0)
        
        # Bearish divergence: new high in price with a lower high in RSI.
        if price_window[-1] >= price_window.max() and rsi_window[-1] < rsi_window.max():
            bearish_div.append(1)
        else:
            bearish_div.append(0)
    
    # Pad the beginning so the arrays match the input length.
    bullish_div = [0] * window + bullish_div
    bearish_div = [0] * window + bearish_div
    return np.array(bullish_div), np.array(bearish_div)

# --- Strategy Class Definition ---

class RSIDivergenceStrategy(Strategy):
    """
    RSI Divergence Strategy:
      - Long entry: Bullish divergence detected and RSI is in oversold territory.
      - Short entry: Bearish divergence detected and RSI is in overbought territory.
    Position sizing is removed; orders are executed with the default trade size.
    """

    # Strategy parameters (adjustable via JSON config)
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
    divergence_window = 5
    atr_period = 14
    atr_multiplier = 2.0

    # Parameter configuration for optimization
    param_config = {
        "rsi_period": {
            "default": 14,
            "range": [10, 14, 20, 25]
        },
        "rsi_overbought": {
            "default": 70,
            "range": [65, 70, 75, 80]
        },
        "rsi_oversold": {
            "default": 30,
            "range": [20, 25, 30, 35]
        },
        "divergence_window": {
            "default": 5,
            "range": [3, 5, 7]
        },
        "atr_period": {
            "default": 14,
            "range": [10, 14, 20, 25]
        },
        "atr_multiplier": {
            "default": 2.0,
            "range": [1.5, 2.0, 2.5, 3.0]
        }
    }

    def init(self):
        # Calculate RSI using our custom implementation.
        self.rsi = self.I(lambda: custom_RSI(self.data.Close, period=self.rsi_period))
        # Calculate ATR using our custom implementation.
        self.atr = self.I(lambda: custom_ATR(self.data.High, self.data.Low, self.data.Close, period=self.atr_period))
        # Identify divergences.
        self.bullish_div, self.bearish_div = self.I(lambda: find_divergences(self.data.Close, self.rsi, window=self.divergence_window))
    
    def next(self):
        # Ensure we have valid indicator values.
        if np.isnan(self.rsi[-1]) or np.isnan(self.atr[-1]):
            return

        atr_val = self.atr[-1]
        entry_price = self.data.Close[-1]

        # Long Entry: Bullish divergence and RSI in oversold territory.
        if not self.position and self.bullish_div[-1] and self.rsi[-1] < self.rsi_oversold:
            stop_loss = entry_price - (atr_val * self.atr_multiplier)
            take_profit = entry_price + (atr_val * self.atr_multiplier * 1.5)
            self.buy(sl=stop_loss, tp=take_profit)

        # Short Entry: Bearish divergence and RSI in overbought territory.
        elif not self.position and self.bearish_div[-1] and self.rsi[-1] > self.rsi_overbought:
            stop_loss = entry_price + (atr_val * self.atr_multiplier)
            take_profit = entry_price - (atr_val * self.atr_multiplier * 1.5)
            self.sell(sl=stop_loss, tp=take_profit)

# Ensure the class is exported.
__all__ = ['RSIDivergenceStrategy']
