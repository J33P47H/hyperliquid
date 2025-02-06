import pandas as pd
import numpy as np
from backtesting import Strategy

class AtrReversionStrategy(Strategy):
    """
    A strategy that uses ATR and Keltner Channels to spot overextended markets 
    and then enters mean-reversion trades with dynamic stop losses based on ATR.
    Uses pandas calculations instead of talib for broader compatibility.
    """
    
    # Strategy parameters
    atr_period = 14
    ema_period = 20
    keltner_mult = 2.0
    stop_loss_atr_mult = 1.5  # Dynamic stop loss multiplier based on ATR

    # Parameter configuration for optimization
    param_config = {
        "atr_period": {
            "default": 14,
            "range": [10, 14, 20, 25]
        },
        "ema_period": {
            "default": 20,
            "range": [10, 20, 30, 40]
        },
        "keltner_mult": {
            "default": 2.0,
            "range": [1.5, 2.0, 2.5, 3.0]
        },
        "stop_loss_atr_mult": {
            "default": 1.5,
            "range": [1.0, 1.5, 2.0, 2.5]
        }
    }

    def init(self):
        # Helper function: Calculate ATR using EMA smoothing
        def calculate_atr(high, low, close, period):
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)
            close_prev = close.shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            tr = tr.replace([np.inf, -np.inf], np.nan).fillna(0)
            atr = tr.ewm(span=period, min_periods=1, adjust=False).mean()
            return atr

        # Helper function: Calculate EMA
        def calculate_ema(prices, period):
            return pd.Series(prices).ewm(span=period, min_periods=1, adjust=False).mean()

        # Register ATR and EMA indicators with the backtesting engine
        self.atr = self.I(lambda: calculate_atr(
            self.data.High, 
            self.data.Low, 
            self.data.Close, 
            self.atr_period
        ), name='atr')

        self.ema = self.I(lambda: calculate_ema(
            self.data.Close,
            self.ema_period
        ), name='ema')

        # Compute Keltner Channels based on the EMA and ATR
        self.keltner_upper = self.I(lambda: self.ema + self.keltner_mult * self.atr)
        self.keltner_lower = self.I(lambda: self.ema - self.keltner_mult * self.atr)

    def next(self):
        # Ensure sufficient data before trading
        if len(self.data) < max(self.atr_period, self.ema_period):
            return

        price = self.data.Close[-1]
        
        # Validate indicator values
        if not (0 < price < float('inf') and 
                0 < self.keltner_upper[-1] < float('inf') and 
                0 < self.keltner_lower[-1] < float('inf') and 
                0 < self.ema[-1] < float('inf')):
            return
        
        # Entry logic for reversion trades without position sizing
        if not self.position:
            # Short setup: price is above the upper Keltner channel and the candle is bearish
            if (price > self.keltner_upper[-1] and 
                self.data.Close[-1] < self.data.Open[-1]):
                
                # Dynamic stop loss: for a short, stop loss is above entry by an ATR multiple
                stop_loss = price + self.stop_loss_atr_mult * self.atr[-1]
                target = self.ema[-1]  # Profit target based on EMA
                
                # Ensure the EMA target is below the entry price for shorts
                if target < price:
                    self.sell(sl=stop_loss, tp=target)

            # Long setup: price is below the lower Keltner channel and the candle is bullish
            elif (price < self.keltner_lower[-1] and 
                  self.data.Close[-1] > self.data.Open[-1]):
                
                # Dynamic stop loss: for a long, stop loss is below entry by an ATR multiple
                stop_loss = price - self.stop_loss_atr_mult * self.atr[-1]
                target = self.ema[-1]  # Profit target based on EMA
                
                # Ensure the EMA target is above the entry price for longs
                if target > price:
                    self.buy(sl=stop_loss, tp=target)

        # Exit logic is handled solely by the stop loss and take profit orders.
