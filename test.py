import numpy as np
import talib

def main():
    # Create a sample data array (e.g., numbers 1 to 20)
    data = np.arange(1, 21, dtype=float)
    print("Input data:", data)

    # Calculate the Simple Moving Average (SMA) with a period of 5
    sma = talib.SMA(data, timeperiod=5)
    print("\nSimple Moving Average (period=5):")
    print(sma)

    # Calculate the Relative Strength Index (RSI) with a period of 14
    # (Note: The first 13 values will be NaN because there isn't enough data)
    rsi = talib.RSI(data, timeperiod=14)
    print("\nRelative Strength Index (RSI, period=14):")
    print(rsi)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred while running TA‑Lib:", e)
