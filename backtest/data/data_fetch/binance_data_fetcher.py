import pandas as pd
import requests
from datetime import datetime
import time

class BinanceFuturesDataFetcher:
    def __init__(self):
        self.base_url = "https://fapi.binance.com"
        self.klines_endpoint = "/fapi/v1/klines"
        
    def get_klines(self, symbol, interval, start_time=None, end_time=None, limit=1000):
        """
        Fetch kline/candlestick data for futures
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')
        interval : str
            Kline interval. Options: 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M
        start_time : str or datetime, optional
            Start time in 'YYYY-MM-DD HH:MM:SS' format or datetime object
        end_time : str or datetime, optional
            End time in 'YYYY-MM-DD HH:MM:SS' format or datetime object
        limit : int, optional
            Number of records to fetch (max 1000)
            
        Returns:
        --------
        pandas.DataFrame
            OHLCV data with timestamp index
        """
        # Convert string dates to timestamps if provided
        if start_time:
            if isinstance(start_time, str):
                start_time = int(datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
            elif isinstance(start_time, datetime):
                start_time = int(start_time.timestamp() * 1000)
                
        if end_time:
            if isinstance(end_time, str):
                end_time = int(datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S').timestamp() * 1000)
            elif isinstance(end_time, datetime):
                end_time = int(end_time.timestamp() * 1000)

        # Prepare parameters
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        # Make request
        response = requests.get(f"{self.base_url}{self.klines_endpoint}", params=params)
        
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.json()}")

        # Convert to DataFrame
        data = response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                                       'taker_buy_quote', 'ignore'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        # Keep only OHLCV columns
        return df[['open', 'high', 'low', 'close', 'volume']]

    def fetch_historical_data(self, symbol, interval, start_time, end_time):
        """
        Fetch historical data by making multiple API calls if needed
        
        Parameters are same as get_klines()
        """
        all_data = []
        current_start = start_time
        
        while True:
            chunk = self.get_klines(symbol, interval, current_start, end_time)
            if chunk.empty:
                break
                
            all_data.append(chunk)
            
            # Update start time for next iteration
            current_start = chunk.index[-1] + pd.Timedelta(seconds=1)
            
            if current_start >= pd.to_datetime(end_time):
                break
                
            # Respect rate limits
            time.sleep(0.5)
        
        if not all_data:
            return pd.DataFrame()
            
        return pd.concat(all_data)

    def fetch_multiple_symbols(self, symbols, interval, start_time, end_time):
        """
        Fetch historical data for multiple symbols
        
        Parameters:
        -----------
        symbols : list
            List of trading pair symbols (e.g., ['BTCUSDT', 'ETHUSDT'])
        interval : str
            Kline interval. Options: 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M
        start_time : str or datetime
            Start time in 'YYYY-MM-DD HH:MM:SS' format or datetime object
        end_time : str or datetime
            End time in 'YYYY-MM-DD HH:MM:SS' format or datetime object
            
        Returns:
        --------
        dict
            Dictionary with symbols as keys and DataFrames as values
        """
        results = {}
        total_symbols = len(symbols)
        
        for i, symbol in enumerate(symbols, 1):
            try:
                print(f"\nFetching data for {symbol} ({i}/{total_symbols})")
                df = self.fetch_historical_data(symbol, interval, start_time, end_time)
                
                if not df.empty:
                    # Save to CSV
                    output_filename = f"data/ohlcv/ohlcv_data_{interval}_{symbol}.csv"
                    # Reset index to make timestamp a column and format it properly
                    df_to_save = df.reset_index()
                    # Rename columns to lowercase for consistency
                    df_to_save.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    df_to_save.to_csv(output_filename, index=False)
                    print(f"Data saved to {output_filename}")
                    print(f"Shape of data: {df.shape}")
                    
                    results[symbol] = df
                else:
                    print(f"No data found for {symbol}")
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue
            
            # Respect rate limits between symbols
            if i < total_symbols:
                time.sleep(1)
        
        return results

# Example usage
if __name__ == "__main__":
    fetcher = BinanceFuturesDataFetcher()
    
    # Example parameters
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]  # Add your symbols here
    interval = "1h"
    start_time = "2025-01-01 00:00:00"
    end_time = datetime.now()
    
    try:
        # Create data directory if it doesn't exist
        from pathlib import Path
        Path("data/ohlcv").mkdir(parents=True, exist_ok=True)
        
        # Fetch data for all symbols
        results = fetcher.fetch_multiple_symbols(symbols, interval, start_time, end_time)
        
        # Print summary
        print("\nSummary:")
        for symbol, df in results.items():
            print(f"\n{symbol}:")
            print(f"Shape: {df.shape}")
            print("First timestamp:", df.index[0])
            print("Last timestamp:", df.index[-1])
        
    except Exception as e:
        print(f"Error: {e}") 