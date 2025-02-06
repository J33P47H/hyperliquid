import pandas as pd
import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover

class EmaVolumeSyncStrategy(Strategy):
    """
    A strategy that uses EMA crossovers with volume confirmation for trade entries.
    Uses pandas calculations instead of talib for broader compatibility.
    """
    
    # Define parameters as class variables first
    ema_period = 20
    volume_ma_period = 20
    
    # Parameter configuration for optimization
    param_config = {
        "ema_period": {
            "default": 20,
            "range": [10, 15, 20, 25, 30]
        },
        "volume_ma_period": {
            "default": 20,
            "range": [10, 15, 20, 25, 30]
        }
    }

    def init(self):
        # Calculate EMAs using pandas
        def calculate_ema(prices, period):
            # Convert to pandas Series and handle invalid values
            prices = pd.Series(prices).replace([np.inf, -np.inf], np.nan)
            # Calculate EMA with minimum periods and fill any remaining NaN
            ema = prices.ewm(span=period, min_periods=1, adjust=False).mean()
            return ema.fillna(prices)

        # Calculate Volume MA using pandas
        def calculate_volume_ma(volume, period):
            # Convert to pandas Series and handle invalid values
            volume = pd.Series(volume).replace([np.inf, -np.inf], np.nan)
            # Calculate MA with minimum periods and fill any remaining NaN
            ma = volume.rolling(window=period, min_periods=1).mean()
            return ma.fillna(volume)

        # Calculate indicators using the wrapper
        prices = self.data.Close
        self.ema = self.I(lambda: calculate_ema(prices, self.ema_period), name='ema')
        self.fast_ema = self.I(lambda: calculate_ema(prices, self.ema_period // 2), name='fast_ema')
        self.volume_ma = self.I(lambda: calculate_volume_ma(self.data.Volume, self.volume_ma_period), name='volume_ma')

    def next(self):
        if len(self.data) < max(self.ema_period, self.volume_ma_period):
            return

        price = self.data.Close[-1]
        current_volume = self.data.Volume[-1]

        # Validate all required values
        if not (0 < price < float('inf') and 
                0 < self.ema[-1] < float('inf') and 
                0 < self.fast_ema[-1] < float('inf') and
                0 < current_volume < float('inf') and
                0 < self.volume_ma[-1] < float('inf')):
            return

        # Trend direction using EMA crossover
        is_uptrend = self.fast_ema[-1] > self.ema[-1]
        is_downtrend = self.fast_ema[-1] < self.ema[-1]

        # Volume confirmation with minimum threshold
        volume_confirmation = current_volume > self.volume_ma[-1] * 1.1  # 10% above average

        # Entry logic for long trades
        if not self.position and is_uptrend and volume_confirmation:
            self.buy()

        # Entry logic for short trades
        elif not self.position and is_downtrend and volume_confirmation:
            self.sell()

        # Exit logic - using trend reversal
        elif self.position:
            if self.position.is_long and is_downtrend:
                self.position.close()
            elif self.position.is_short and is_uptrend:
                self.position.close() 