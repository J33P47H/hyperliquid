import pandas as pd
import numpy as np
from backtesting import Strategy

class AtrReversionStrategy(Strategy):
    """
    A strategy that uses ATR and Keltner Channels to spot overextended markets and then enters mean-reversion trades.
    Uses pandas calculations instead of talib for broader compatibility.
    """
    
    # Define parameters as class variables first
    atr_period = 14
    ema_period = 20
    keltner_mult = 2.0
    risk_percent = 0.01
    trade_fraction = 0.5
    
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
        "risk_percent": {
            "default": 0.01,
            "range": [0.005, 0.01, 0.015, 0.02]
        },
        "trade_fraction": {
            "default": 0.5,
            "range": [0.25, 0.5, 0.75, 1.0]
        }
    }

    def init(self):
        # Calculate ATR
        def calculate_atr(high, low, close, period):
            # Convert inputs to pandas Series
            high = pd.Series(high)
            low = pd.Series(low)
            close = pd.Series(close)
            close_prev = close.shift(1)
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)
            
            # True Range is the maximum of the three
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Replace any invalid values with 0
            tr = tr.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Calculate ATR using EMA for more stability
            atr = tr.ewm(span=period, min_periods=1, adjust=False).mean()
            return atr

        # Calculate EMA
        def calculate_ema(prices, period):
            return pd.Series(prices).ewm(span=period, min_periods=1, adjust=False).mean()

        # Calculate indicators using the wrapper
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

        # Calculate Keltner Channels
        self.keltner_upper = self.I(lambda: self.ema + self.keltner_mult * self.atr)
        self.keltner_lower = self.I(lambda: self.ema - self.keltner_mult * self.atr)

    def next(self):
        if len(self.data) < max(self.atr_period, self.ema_period):
            return

        price = self.data.Close[-1]
        
        # Validate all required values
        if not (0 < price < float('inf') and 
                0 < self.keltner_upper[-1] < float('inf') and 
                0 < self.keltner_lower[-1] < float('inf') and 
                0 < self.ema[-1] < float('inf')):
            return
        
        # Entry logic for potential reversion trades
        if not self.position:
            # Short setup: price closed above upper Keltner and current candle is bearish
            if (price > self.keltner_upper[-1] and 
                self.data.Close[-1] < self.data.Open[-1]):
                
                stop_loss = price * 1.02  # 2% above entry
                risk_per_unit = stop_loss - price
                
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_percent * self.trade_fraction
                    position_size = int(risk_amount / risk_per_unit)
                    if position_size > 0:
                        target = self.ema[-1]  # Target the EMA
                        if target < price:  # Ensure target is below entry for shorts
                            self.sell(size=position_size, sl=stop_loss, tp=target)

            # Long setup: price closed below lower Keltner and current candle is bullish
            elif (price < self.keltner_lower[-1] and 
                  self.data.Close[-1] > self.data.Open[-1]):
                
                stop_loss = price * 0.98  # 2% below entry
                risk_per_unit = price - stop_loss
                
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_percent * self.trade_fraction
                    position_size = int(risk_amount / risk_per_unit)
                    if position_size > 0:
                        target = self.ema[-1]  # Target the EMA
                        if target > price:  # Ensure target is above entry for longs
                            self.buy(size=position_size, sl=stop_loss, tp=target)

        # Exit logic - using stop loss and take profit only
        # No additional exit conditions needed as we're using strict SL/TP 