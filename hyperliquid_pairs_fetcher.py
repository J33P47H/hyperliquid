import requests
import json
from datetime import datetime

def fetch_hyperliquid_pairs():
    """
    Fetch all perpetual trading pairs from Hyperliquid exchange.
    Returns a list of dictionaries containing pair information.
    """
    try:
        # Hyperliquid API endpoint for meta information
        url = "https://api.hyperliquid.xyz/info"
        
        # Request data
        payload = {
            "type": "meta"
        }
        
        # Make POST request
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse response
        data = response.json()
        
        # Extract and format pairs information
        pairs = []
        for coin in data.get("universe", []):
            pair_info = {
                "symbol": f"{coin}USD",
                "base_currency": coin,
                "quote_currency": "USD",
                "timestamp": datetime.now().isoformat()
            }
            pairs.append(pair_info)
        
        return pairs
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def main():
    # Fetch pairs
    pairs = fetch_hyperliquid_pairs()
    
    if pairs:
        print(f"\nFound {len(pairs)} perpetual pairs on Hyperliquid:")
        print("-" * 50)
        
        # Print pairs in a formatted way
        for pair in pairs:
            print(f"Symbol: {pair['symbol']}")
            print(f"Base Currency: {pair['base_currency']}")
            print(f"Quote Currency: {pair['quote_currency']}")
            print(f"Timestamp: {pair['timestamp']}")
            print("-" * 50)
    else:
        print("Failed to fetch pairs from Hyperliquid.")

if __name__ == "__main__":
    main() 