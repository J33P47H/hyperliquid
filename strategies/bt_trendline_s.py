import numpy as np
import pandas as pd
from backtesting import Strategy, Backtest

# -------------------------------------------------------------------
# Helper Functions (from the trendline optimization algorithm)
# -------------------------------------------------------------------
def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    # Calculate the intercept of the line going through the pivot point
    intercept = -slope * pivot + y[pivot]
    # Get the line values across the data series
    line_vals = slope * np.arange(len(y)) + intercept
    # Differences between the line and the actual price
    diffs = line_vals - y

    # For a support line, if any difference is positive the line is above price (invalid)
    if support and diffs.max() > 1e-5:
        return -1.0
    # For a resistance line, if any difference is negative the line is below price (invalid)
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Return the sum of squared differences as an error metric
    err = (diffs ** 2.0).sum()
    return err

def optimize_slope(support: bool, pivot: int, init_slope: float, y: np.array):
    slope_unit = (y.max() - y.min()) / len(y)
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step

    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert(best_err >= 0.0)

    get_derivative = True
    derivative = None
    while curr_step > min_step:
        if get_derivative:
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err
            if test_err < 0.0:
                raise Exception("Derivative failed. Check your data.")
            get_derivative = False

        if derivative > 0.0:
            test_slope = best_slope - slope_unit * curr_step
        else:
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            curr_step *= 0.5  # reduce step size if no improvement
        else:
            best_err = test_err 
            best_slope = test_slope
            get_derivative = True

    return best_slope, -best_slope * pivot + y[pivot]

def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)  # linear fit to the close price
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (high - line_points).argmax() 
    lower_pivot = (low - line_points).argmin() 
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)
    return support_coefs, resist_coefs

# -------------------------------------------------------------------
# Strategy Class Using Full Equity Exposure for Benchmarking
# -------------------------------------------------------------------
class TrendLineStrategy(Strategy):
    """
    A strategy that uses dynamically computed support and resistance trendlines
    (via an optimization algorithm) to trade retests of these zones.
    """
    # Parameters for trendline computation and signal tolerance
    lookback = 30
    tolerance = 0.5  # Adjust as necessary for your asset's price scale.
    risk_reward = 25  # Effective risk/reward ratio = risk_reward/10 (e.g., 25 -> 2.5:1)

    # Parameter configuration for optimization
    param_config = {
        "lookback": {
            "default": 30,
            "range": [20, 30, 40, 50]
        },
        "tolerance": {
            "default": 0.5,
            "range": [0.3, 0.5, 0.7, 1.0]
        },
        "risk_reward": {
            "default": 25,
            "range": [15, 20, 25, 30]  # Corresponds to 1.5:1, 2:1, 2.5:1, 3:1
        }
    }

    def init(self):
        # Set instance parameters from class parameters or defaults
        self.lookback = getattr(self, 'lookback', 30)
        self.tolerance = getattr(self, 'tolerance', 0.5)
        self.risk_reward = getattr(self, 'risk_reward', 25)

    def next(self):
        # Ensure we have enough data for the lookback window.
        if len(self.data) < self.lookback:
            return

        # Get the recent lookback window data as numpy arrays.
        high = np.array(self.data.High[-self.lookback:])
        low = np.array(self.data.Low[-self.lookback:])
        close = np.array(self.data.Close[-self.lookback:])

        # Compute dynamic support and resistance trendlines.
        support_coefs, resist_coefs = fit_trendlines_high_low(high, low, close)
        x_val = self.lookback - 1  # Evaluate at the end of the window.
        support_level = support_coefs[0] * x_val + support_coefs[1]
        resist_level = resist_coefs[0] * x_val + resist_coefs[1]
        current_price = self.data.Close[-1]

        # Effective risk/reward multiplier.
        eff_risk_reward = self.risk_reward / 10.0

        # If not already in a position, check for entry signals.
        if not self.position:
            # LONG ENTRY: If current price is within tolerance of the computed support level.
            if abs(current_price - support_level) < self.tolerance:
                # Set stop loss slightly below the support level.
                stop_loss = support_level * 0.99
                risk_per_unit = current_price - stop_loss
                if risk_per_unit > 0:
                    target = current_price + risk_per_unit * eff_risk_reward
                    if 0 < target < float('inf'):
                        self.buy(sl=stop_loss, tp=target)
            # SHORT ENTRY: If current price is within tolerance of the computed resistance level.
            elif abs(current_price - resist_level) < self.tolerance:
                # Set stop loss slightly above the resistance level.
                stop_loss = resist_level * 1.01
                risk_per_unit = stop_loss - current_price
                if risk_per_unit > 0:
                    target = current_price - risk_per_unit * eff_risk_reward
                    if 0 < target < float('inf'):
                        self.sell(sl=stop_loss, tp=target)