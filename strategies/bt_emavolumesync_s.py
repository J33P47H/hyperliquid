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
    risk_per_trade = 0.01
    risk_reward_ratio = 2
    
    # Parameter configuration for optimization
    param_config = {
        "ema_period": {
            "default": 20,
            "range": [10, 15, 20, 25, 30]
        },
        "volume_ma_period": {
            "default": 20,
            "range": [10, 15, 20, 25, 30]
        },
        "risk_per_trade": {
            "default": 0.01,
            "range": [0.005, 0.01, 0.015, 0.02]
        },
        "risk_reward_ratio": {
            "default": 2,
            "range": [1.5, 2.0, 2.5, 3.0]
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
            # Calculate position size based on risk
            stop_loss = price * 0.98  # 2% below entry
            risk_per_unit = price - stop_loss
            
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_per_trade
                position_size = risk_amount / risk_per_unit
                position_size = max(1, round(position_size))

                # Calculate take profit based on risk-reward ratio
                take_profit = price + (risk_per_unit * self.risk_reward_ratio)
                
                # Validate target price
                if 0 < take_profit < float('inf'):
                    self.buy(size=position_size, sl=stop_loss, tp=take_profit)

        # Entry logic for short trades
        elif not self.position and is_downtrend and volume_confirmation:
            # Calculate position size based on risk
            stop_loss = price * 1.02  # 2% above entry
            risk_per_unit = stop_loss - price
            
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_per_trade
                position_size = risk_amount / risk_per_unit
                position_size = max(1, round(position_size))

                # Calculate take profit based on risk-reward ratio
                take_profit = price - (risk_per_unit * self.risk_reward_ratio)
                
                # Validate target price
                if 0 < take_profit < float('inf'):
                    self.sell(size=position_size, sl=stop_loss, tp=take_profit)

        # Exit logic - using stop loss and take profit only
        # No additional exit conditions needed as we're using strict SL/TP 