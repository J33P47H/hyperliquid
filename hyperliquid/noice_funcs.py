###noice funcs
import dontshare as d
from eth_account.signers.local import LocalAccount
import eth_account
import json
import time
import pandas as pd
from hyperliquid.info import Info 
from hyperliquid.exchange import Exchange 
from hyperliquid.utils import constants 
import ccxt
from datetime import datetime, timedelta
import requests
import schedule

symbol = 'CRV'
timeframe = '4h'
coin = symbol 
secret_key = d.private_key
account: LocalAccount = eth_account.Account.from_key(secret_key)

def acct_bal(account):

    # account = LocalAccount = eth_account.Account.from_key(key)
    info = Info(constants.MAINNET_API_URL)
    user_state = info.user_state(account.address)

    print(f'this is current account value: {user_state["marginSummary"]["accountValue"]}')

    acct_value = user_state["marginSummary"]["accountValue"]

    return acct_value

def ask_bid(symbol):
	#this gets the ask and bid for any symbol passed in
	
	url = 'https://api.hyperliquid.xyz/info'
	headers = {'Content-Type': 'application/json'}
	
	data = {
		'type': "l2Book",
		'coin': symbol
	}
	
	try:
		response = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
		l2_data = response.json()
		l2_data = l2_data['levels']
		
		#get ask bid
		bid = float(l2_data[0][0]['px'])
		ask = float(l2_data[1][0]['px'])
		return ask, bid, l2_data
	except requests.exceptions.Timeout:
		print("Timeout getting ask/bid prices")
		return 0, 0, []
	except Exception as e:
		print(f"Error getting ask/bid prices: {e}")
		return 0, 0, []

##FUNCTION: GET SIZE PRICE DECIMALS
##Returns size decimals, price decimals
def get_sz_px_decimals(coin):
    #this returns size and price decimals
    
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': "application/json"}
    data = {'type': "meta"}
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        data = response.json()
        symbols = data['universe']
        symbol_info = next((s for s in symbols if s['name'] == coin), None)
        if symbol_info:
            sz_decimals = symbol_info['szDecimals']
        else:
            print('symbol not found')
            return 0, 0
    else:
        print('Error:', response.status_code)
        return 0, 0
    
    ask = ask_bid(coin)[0]
    ask_str = str(ask)
    if ',' in ask_str:
        px_decimals = len(ask_str.split('.')[1])
    else:
        px_decimals = 0
    
    print(f'{coin} this is the price {sz_decimals} decimals')
    
    return sz_decimals, px_decimals

def adjust_leverage_size_signal(symbol, leverage, account):
	
	print('leverage', leverage)
	
	# Initialize Exchange and Info
	exchange = Exchange(base_url=constants.MAINNET_API_URL, wallet=account)
	info = Info(constants.MAINNET_API_URL)
	
	# Get metadata to check valid leverage with timeout
	url = 'https://api.hyperliquid.xyz/info'
	headers = {'Content-Type': "application/json"}
	data = {'type': "meta"}
	try:
		response = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
		meta_data = response.json()
		symbol_info = next((s for s in meta_data['universe'] if s['name'] == symbol), None)
		
		if symbol_info:
			max_leverage = symbol_info['maxLeverage']
			if leverage > max_leverage:
				print(f"Warning: Leverage {leverage} exceeds maximum allowed {max_leverage}. Setting to max.")
				leverage = max_leverage
	except requests.exceptions.Timeout:
		print("Timeout getting metadata, proceeding with provided leverage")
	except Exception as e:
		print(f"Error getting metadata: {e}, proceeding with provided leverage")
	
	try:
		user_state = info.user_state(account.address)
		acct_value = float(user_state["marginSummary"]["accountValue"])
		print(f'Account value: ${acct_value}')
		
		acct_val95 = acct_value * 0.95  # Using 95% of account value
		print(f'Using 95% of account value: ${acct_val95}')
		
		leverage_result = exchange.update_leverage(leverage, symbol)
		print(f'Leverage update result: {leverage_result}')
		
		current_price = ask_bid(symbol)[0]
		print(f'Current {symbol} price: ${current_price}')
		
		# Calculate position size in USD first
		position_size_usd = acct_val95 * leverage
		
		# Convert USD size to token amount
		size = position_size_usd / current_price
		
		rounding = get_sz_px_decimals(symbol)[0]
		size = round(size, rounding)
		print(f'Position size: {size} {symbol} (${position_size_usd:.2f} notional)')
		
		return leverage, size
	
	except Exception as e:
		print(f"Error during execution: {e}")
		return leverage, 0  # Return 0 size in case of error
	finally:
		# Clean up connections
		if hasattr(exchange, 'close'):
			exchange.close()
		if hasattr(info, 'close'):
			info.close()
	

##FUNCTION : ENTER LIMIT ORDER
##Returns Order result
def limit_order(coin, is_buy, sz, limit_px, reduce_only, account):
    
    exchange = Exchange(account, constants.MAINNET_API_URL)
    
    rounding = get_sz_px_decimals(coin)[0]
    sz = round(sz, rounding)
    
    print(f'coin: {coin}, type: {type(coin)}')
    print(f'is_buy: {is_buy}, type: {type(coin)}')
    print(f'sz: {sz}, type: {type(limit_px)}')
    print(f'reduce_only: {reduce_only}, type: {type(reduce_only)}')
    print(f'placing limit order for {coin} {sz} @ {limit_px}')
    
    order_result = exchange.order(coin, is_buy, sz, limit_px, {'limit':{'tif': "Gtc"}}, reduce_only=reduce_only)
    
    if is_buy == True:
        print(f'limit BUY order placed, resting: {order_result["response"]["data"]["statuses"][0]}')
    else:
        print(f'limit SELL order placed,resting: {order_result["response"]["data"]["statuses"][0]}')
    
    return order_result

def cancel_all_orders(account):
	exchange = Exchange(account, constants.MAINNET_API_URL)
	info = Info(constants.MAINNET_API_URL, skip_ws=True)
	
	open_orders = info.open_orders(account.address)
	
	print('above are the open orders... need to cancel any...')
	for open_order in open_orders:
		print(f'cancelling order {open_order}')
		exchange.cancel(open_order['coin'], open_order['oid'])
		

def process_data_to_df(snapshot_data, write_to_csv=False):
    """
    Process snapshot data into a DataFrame and optionally write to CSV
    
    Args:
        snapshot_data: The OHLCV data to process
        write_to_csv (bool): If True, saves the DataFrame to a CSV file
    
    Returns:
        pd.DataFrame: The processed DataFrame
    """
    if not snapshot_data:
        return pd.DataFrame()
        
    #Assuming the response contains a list of candles
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    data = []
    for snapshot in snapshot_data:
        timestamp = datetime.fromtimestamp(snapshot['t'] / 1000).strftime('%Y-%m-%d %H-%M-%S')
        open_price = snapshot['o']
        high_price = snapshot['h']
        low_price = snapshot['l']
        close_price = snapshot['c']
        volume = snapshot['v']
        data.append([timestamp, open_price, high_price, low_price, close_price, volume])
    
    df = pd.DataFrame(data, columns=columns)
    
    #Calculate support and resistance, excluding the last two rows for the calculation
    if len(df) > 2:
        df['support'] = df[:-2]['close'].min()
        df['resis'] = df[:-2]['close'].max()
    else:
        df['support'] = df['close'].min()
        df['resis'] = df['close'].max()
    
    if write_to_csv:
        # Create filename with current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'ohlcv_data_{timestamp}.csv'
        df.to_csv(filename, index=False)
        print(f'Data written to {filename}')
    
    return df

def get_ohlcv(symbol, interval, lookback_days):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': "application/json"}
    data = {
        'type': "candleSnapshot",
        'req':{
            'coin': symbol,
            'interval': interval,
            'startTime': int(start_time.timestamp()*1000),
            'endTime': int(end_time.timestamp()*1000)
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        snapshot_data = response.json()
        return snapshot_data
    else:
        print(f'Error fetching data for {symbol}: {response.status_code}')
        return None
    


def fetch_candle_snapshot(symbol, interval, start_time, end_time):

    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': "application/json"}
    data = {
        'type': "candleSnapshot",
        'req': {
            'coin': symbol,
            'interval': interval,
            'startTime': int(start_time.timestamp()*1000),
            'endTime': int(end_time.timestamp()*1000)
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        snapshot_data = response.json()
        return snapshot_data
    

def get_position(symbol, account):
    
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    
    print(f'this is current account value: {user_state["marginSummary"]["accountValue"]}')
    
    # Initialize all variables
    positions = []
    in_pos = False
    size = 0
    pos_sym = None
    entry_px = 0
    pnl_perc = 0
    long = None
    
    print(f'this is the symbol {symbol}')
    
    for position in user_state["assetPositions"]:
        if (position["position"]["coin"] == symbol) and float(position["position"]["szi"]) != 0:
            positions.append(position["position"])
            in_pos = True
            size = float(position["position"]["szi"])
            pos_sym = position["position"]["coin"]
            entry_px = float(position["position"]["entryPx"])  # Fixed variable name
            pnl_perc = float(position["position"]["returnOnEquity"]) * 100
            print(f'this is the pnl perc {pnl_perc}')
            break
    
    if size > 0:
        long = True
    elif size < 0:
        long = False
    
    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long


##FUNCTION : KILL SWITCH
##Returns None
def kill_switch(symbol, account):
	
    position, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
	
    while im_in_pos == True:
        cancel_all_orders(account)
        ask, bid, l2 = ask_bid(symbol)
        pos_size = abs(pos_size)
        
        if long == True:
            limit_order(pos_sym, False, pos_size, ask, True, account)
            print('kill switch - SELL TO CLOSE SUBMITTED ')
            time.sleep(5)
        elif long == False:
            limit_order(pos_sym, True, pos_size, bid, True, account)
            print('kill switch - BUY TO CLOSE SUBMITTED ')
            time.sleep(5)
            
        position, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
    
    print('position successfully closed in the kill switch')

def pnl_close(symbol, target, max_loss, account):
    
    print('starting pnl close')
    
    position, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
    
    if im_in_pos:
        # Get current price
        current_price = ask_bid(symbol)[0] if long else ask_bid(symbol)[1]
        
        # Get leverage from position
        leverage = float(position[0]['leverage']['value']) if position else 0
        
        print(f'Current price: ${current_price:.5f}')
        print(f'Entry price: ${entry_px:.5f}')
        print(f'Leverage: {leverage}x')
        print(f'ROE (leveraged PnL): {pnl_perc:.3f}%')
        
        if pnl_perc > target:
            print(f'ROE gain is {pnl_perc:.3f}% and target is {target}%... closing position = WIN')
            kill_switch(pos_sym, account)
        elif pnl_perc <= max_loss:
            print(f'ROE loss is {pnl_perc:.3f}% and max loss is {max_loss}%... closing position = LOSS')
            kill_switch(pos_sym, account)
        else:
            print(f'ROE is {pnl_perc:.3f}%, max loss is {max_loss}% and target is {target}%... not closing position')
    
    print('finished with pnl close')


def get_position_and_max_position(symbol, account, max_positions):
    
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    
    print(f'this is current account value: {user_state["marginSummary"]["accountValue"]}')
    
    positions = []
    open_positions = []
    in_pos = False
    size = 0
    pos_sym = None
    entry_px = 0
    pnl_perc = 0
    long = None
    
    # CHECKING MAX POSITIONS FIRST
    # Iterate over each position in the assetPositions list
    for position in user_state["assetPositions"]:
        # Check if the position size ["szi"] is not zero, indicating an open position
        if float(position["position"]["szi"]) != 0:
            # If it's an open position, add the coin symbol to the open_positions list
            open_positions.append(position["position"]["coin"])
    
    print(open_positions)
    num_of_pos = len(open_positions)
    
    if len(open_positions) > max_positions:
        print(f'we are in {len(open_positions)} positions and max pos is {max_positions}... closing positions...')
        # Close all positions that exceed the maximum
        for position in open_positions:
            kill_switch(position, account)
    else:
        print(f'we are in {len(open_positions)} positions and max pos is {max_positions}... not closing positions')
    
    # Check for specific symbol position
    for position in user_state["assetPositions"]:
        if (position["position"]["coin"] == symbol) and float(position["position"]["szi"]) != 0:
            positions.append(position["position"])
            in_pos = True
            size = float(position["position"]["szi"])
            pos_sym = position["position"]["coin"]
            entry_px = float(position["position"]["entryPx"])
            pnl_perc = float(position["position"]["returnOnEquity"]) * 100
            print(f'this is the pnl perc {pnl_perc}')
            break
    
    if size > 0:
        long = True
    elif size < 0:
        long = False
    
    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long, num_of_pos


def close_all_positions(account):
    
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    print(f'this is the current account value: {user_state["marginSummary"]["accountValue"]}')
    
    positions = []
    open_positions = []
    
    print("Checking all positions...")
    print(user_state["assetPositions"])
    
    #cancel all orders
    cancel_all_orders(account)
    print('all orders have been cancelled')
    
    for position in user_state["assetPositions"]:
        if float(position["position"]["szi"]) != 0:
            open_positions.append(position["position"]["coin"])
            
    for position in open_positions:
        kill_switch(position, account)
    
    print('all positions have been closed')