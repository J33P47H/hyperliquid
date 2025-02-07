### noice_funcs.py
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

# --- RICH IMPORTS ---
from rich.console import Console
from rich.table import Table
from rich import box

# Initialize a global console instance for colorful prints
console = Console()

symbol = 'CRV'
timeframe = '4h'
coin = symbol 
secret_key = d.private_key
account: LocalAccount = eth_account.Account.from_key(secret_key)


def acct_bal(account):
    """
    Fetches and returns the account value (marginSummary.accountValue) from Hyperliquid.
    """
    info = Info(constants.MAINNET_API_URL)
    user_state = info.user_state(account.address)
    acct_value = user_state["marginSummary"]["accountValue"]

    console.print(
        f"[bright_white]Account value:[/bright_white] [bold green]{acct_value}[/bold green]"
    )
    return acct_value


def ask_bid(symbol):
    """
    Gets the ask and bid for any symbol via Hyperliquid's L2 orderbook.
    Returns (ask, bid, full_l2_data).
    """
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
        
        bid = float(l2_data[0][0]['px'])
        ask = float(l2_data[1][0]['px'])
        
        console.print(
            f"[bold magenta]{symbol}[/bold magenta] "
            f"Bid: [bold cyan]{bid}[/bold cyan], Ask: [bold cyan]{ask}[/bold cyan]"
        )
        return ask, bid, l2_data
    except requests.exceptions.Timeout:
        console.print(f"[bold red]Timeout[/bold red] getting ask/bid prices for {symbol}")
        return 0, 0, []
    except Exception as e:
        console.print(f"[bold red]Error[/bold red] getting ask/bid prices for {symbol}: {e}")
        return 0, 0, []


def get_sz_px_decimals(coin):
    """
    Returns size decimals and price decimals for a given coin from Hyperliquid metadata.
    """
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
            console.print(f"[bold red]Symbol {coin} not found in metadata[/bold red]")
            return 0, 0
    else:
        console.print(f"[bold red]Error[/bold red]: HTTP {response.status_code} fetching metadata")
        return 0, 0
    
    ask_price = ask_bid(coin)[0]
    ask_str = str(ask_price)
    if '.' in ask_str:
        px_decimals = len(ask_str.split('.')[1])
    else:
        px_decimals = 0
    
    console.print(
        f"[bold white]{coin}[/bold white] decimals => "
        f"[green]size decimals:[/green] {sz_decimals}, [green]price decimals:[/green] {px_decimals}"
    )
    
    return sz_decimals, px_decimals


def adjust_leverage_size_signal(symbol, leverage, account):
    """
    Updates leverage for the given symbol and calculates a position size
    using ~95% of the account value times the requested leverage.
    Returns (final_leverage, size).
    """
    console.print(f"[bold yellow]Adjusting leverage[/bold yellow] -> {leverage}x for {symbol}")
    
    exchange = Exchange(base_url=constants.MAINNET_API_URL, wallet=account)
    info = Info(constants.MAINNET_API_URL)

    # Grab metadata to ensure we don't exceed max leverage
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
                console.print(
                    f"[bold red]Warning[/bold red]: requested leverage {leverage} > max {max_leverage}. "
                    "Setting to max."
                )
                leverage = max_leverage
    except requests.exceptions.Timeout:
        console.print("[bold red]Timeout[/bold red] getting metadata, proceeding with provided leverage")
    except Exception as e:
        console.print(f"[bold red]Error[/bold red] getting metadata: {e}, proceeding with provided leverage")
    
    try:
        user_state = info.user_state(account.address)
        acct_value = float(user_state["marginSummary"]["accountValue"])
        
        console.print(f"[white]Account value[/white]: [bold cyan]${acct_value:.2f}[/bold cyan]")
        
        acct_val95 = acct_value * 0.95
        console.print(
            f"Using [yellow]95%[/yellow] of account value => "
            f"[bold magenta]${acct_val95:.2f}[/bold magenta]"
        )
        
        leverage_result = exchange.update_leverage(leverage, symbol)
        console.print(
            f"[green]Leverage update result[/green]: "
            f"{leverage_result}"
        )
        
        current_price = ask_bid(symbol)[0]
        console.print(
            f"[bold]Current {symbol} price[/bold]: [cyan]${current_price:.2f}[/cyan]"
        )
        
        # Calculate position size
        position_size_usd = acct_val95 * leverage
        size = position_size_usd / current_price
        
        rounding = get_sz_px_decimals(symbol)[0]
        size = round(size, rounding)
        
        console.print(
            f"[bold blue]Position size[/bold blue]: {size} {symbol} "
            f"(Notional ~ [bold green]${position_size_usd:.2f}[/bold green])"
        )
        
        return leverage, size
    
    except Exception as e:
        console.print(f"[bold red]Error[/bold red] during execution: {e}")
        return leverage, 0
    finally:
        if hasattr(exchange, 'close'):
            exchange.close()
        if hasattr(info, 'close'):
            info.close()


def limit_order(coin, is_buy, sz, limit_px, reduce_only, account):
    """
    Places a limit order (GTC) on Hyperliquid.
    Returns the order result (dict) from the exchange.
    """
    exchange = Exchange(account, constants.MAINNET_API_URL)
    rounding = get_sz_px_decimals(coin)[0]
    sz = round(sz, rounding)
    
    console.print(
        f"[bold white]Placing limit order[/bold white] => "
        f"Coin: [magenta]{coin}[/magenta], [cyan]{'BUY' if is_buy else 'SELL'}[/cyan], "
        f"Size: [green]{sz}[/green], Price: [yellow]{limit_px}[/yellow], "
        f"reduce_only: [red]{reduce_only}[/red]"
    )
    
    order_result = exchange.order(
        coin, is_buy, sz, limit_px,
        {'limit': {'tif': "Gtc"}},
        reduce_only=reduce_only
    )
    
    # Print status
    side_str = "BUY" if is_buy else "SELL"
    status = order_result["response"]["data"]["statuses"][0]
    console.print(
        f"[bold]{side_str} order[/bold] placed => "
        f"[bold magenta]{status}[/bold magenta]"
    )
    
    return order_result


def cancel_all_orders(account):
    """
    Cancels all open orders for the account.
    """
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    
    open_orders = info.open_orders(account.address)
    console.print(f"[bold red]Cancelling all open orders[/bold red] => found {len(open_orders)} orders.")
    
    for open_order in open_orders:
        console.print(f"...cancelling order [yellow]{open_order}[/yellow]")
        exchange.cancel(open_order['coin'], open_order['oid'])


def process_data_to_df(snapshot_data, write_to_csv=False):
    """
    Process snapshot data (candles) into a DataFrame.
    Optionally writes to CSV if write_to_csv=True.
    Adds 'support' and 'resis' columns based on historical min/max close.
    """
    if not snapshot_data:
        console.print("[bold red]No snapshot data to process[/bold red]")
        return pd.DataFrame()
        
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
    
    # Support and resistance logic
    if len(df) > 2:
        df['support'] = df[:-2]['close'].min()
        df['resis'] = df[:-2]['close'].max()
    else:
        df['support'] = df['close'].min()
        df['resis'] = df['close'].max()
    
    if write_to_csv:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'ohlcv_data_{ts}.csv'
        df.to_csv(filename, index=False)
        console.print(f"[bold green]Data written to {filename}[/bold green]")
    
    console.print(f"[bold cyan]DataFrame created[/bold cyan] => {df.shape[0]} rows")
    return df


def get_ohlcv(symbol, interval, lookback_days):
    """
    Fetches OHLCV data from Hyperliquid's candleSnapshot endpoint for the specified symbol,
    interval, and number of days lookback.
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': "application/json"}
    data = {
        'type': "candleSnapshot",
        'req': {
            'coin': symbol,
            'interval': interval,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000)
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        snapshot_data = response.json()
        console.print(f"[white]Fetched OHLCV[/white] for [bold]{symbol} {interval}[/bold], days={lookback_days}")
        return snapshot_data
    else:
        console.print(f"[bold red]Error[/bold red] fetching data for {symbol}: HTTP {response.status_code}")
        return None


def fetch_candle_snapshot(symbol, interval, start_time, end_time):
    """
    Fetches a candleSnapshot for a given symbol & interval between start_time and end_time.
    """
    url = 'https://api.hyperliquid.xyz/info'
    headers = {'Content-Type': "application/json"}
    data = {
        'type': "candleSnapshot",
        'req': {
            'coin': symbol,
            'interval': interval,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000)
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        console.print(f"[green]Fetched custom candle snapshot[/green] for {symbol} {interval}.")
        return response.json()
    else:
        console.print(f"[red]Error[/red] fetching custom candle snapshot => HTTP {response.status_code}")


def get_position(symbol, account):
    """
    Gets the current position (if any) for the given symbol.
    Returns a tuple:
        (positions, in_pos, size, pos_sym, entry_px, pnl_perc, long_bool)
    """
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    
    console.print(
        f"[bold yellow]Fetching position[/bold yellow] for [magenta]{symbol}[/magenta] "
        f"|| acctValue: [bold green]{user_state['marginSummary']['accountValue']}[/bold green]"
    )
    
    positions = []
    in_pos = False
    size = 0
    pos_sym = None
    entry_px = 0
    pnl_perc = 0
    long_pos = None
    
    for position in user_state["assetPositions"]:
        if (position["position"]["coin"] == symbol) and float(position["position"]["szi"]) != 0:
            positions.append(position["position"])
            in_pos = True
            size = float(position["position"]["szi"])
            pos_sym = position["position"]["coin"]
            entry_px = float(position["position"]["entryPx"])
            pnl_perc = float(position["position"]["returnOnEquity"]) * 100
            console.print(f"[bold cyan]PnL %[/bold cyan]: {pnl_perc:.2f}%")
            break
    
    if size > 0:
        long_pos = True
    elif size < 0:
        long_pos = False
    
    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long_pos


def kill_switch(symbol, account):
    """
    Immediately closes (or attempts to close) any open position in the specified symbol
    by canceling all orders and placing an offsetting limit order.
    """
    console.print(f"[bold red]KILL SWITCH[/bold red] => [yellow]{symbol}[/yellow] position will be closed!")
    position, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long_pos = get_position(symbol, account)
    
    while im_in_pos:
        cancel_all_orders(account)
        ask, bid, l2 = ask_bid(symbol)
        abs_size = abs(pos_size)
        
        if long_pos:
            limit_order(pos_sym, False, abs_size, ask, True, account)
            console.print("[red]Kill switch => SELL TO CLOSE submitted[/red]")
            time.sleep(5)
        else:
            limit_order(pos_sym, True, abs_size, bid, True, account)
            console.print("[red]Kill switch => BUY TO CLOSE submitted[/red]")
            time.sleep(5)
        
        position, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long_pos = get_position(symbol, account)
    
    console.print("[bold green]Position successfully closed via kill switch[/bold green]")


def pnl_close(symbol, target, max_loss, account):
    """
    Checks if the current position's ROE is above `target`% or below `max_loss`%.
    If so, kill_switch to close position.
    """
    console.print(f"[bold blue]PNL CLOSE check[/bold blue] => Target: {target}%, Max Loss: {max_loss}%")
    
    position, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long_pos = get_position(symbol, account)
    if im_in_pos:
        current_price = ask_bid(symbol)[0] if long_pos else ask_bid(symbol)[1]
        console.print(
            f"[white]Current price:[/white] [bold cyan]{current_price:.4f}[/bold cyan], "
            f"[white]Entry price:[/white] [bold cyan]{entry_px:.4f}[/bold cyan], "
            f"ROE: [bold magenta]{pnl_perc:.2f}%[/bold magenta]"
        )
        
        if pnl_perc > target:
            console.print(f"[green]ROE {pnl_perc:.2f}% exceeds target {target}% => closing position[/green]")
            kill_switch(pos_sym, account)
        elif pnl_perc <= max_loss:
            console.print(f"[red]ROE {pnl_perc:.2f}% below max loss {max_loss}% => closing position[/red]")
            kill_switch(pos_sym, account)
        else:
            console.print(
                f"[yellow]ROE {pnl_perc:.2f}% is within the hold range "
                f"({max_loss}% ~ {target}%), not closing.[/yellow]"
            )
    else:
        console.print("[cyan]No open position, nothing to close.[/cyan]")


def get_position_and_max_position(symbol, account, max_positions):
    """
    Fetch open positions for the account, enforces a max_positions rule globally,
    closes positions if the total open positions exceed max_positions.
    Then returns the position info for the specified symbol.
    """
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    
    console.print(f"[bold yellow]Enforcing max positions[/bold yellow]: {max_positions}")
    console.print(
        f"Account value => [green]{user_state['marginSummary']['accountValue']}[/green]"
    )
    
    positions = []
    open_positions = []
    in_pos = False
    size = 0
    pos_sym = None
    entry_px = 0
    pnl_perc = 0
    long_pos = None
    
    # Check how many positions are open
    for pos in user_state["assetPositions"]:
        if float(pos["position"]["szi"]) != 0:
            open_positions.append(pos["position"]["coin"])
    
    num_of_pos = len(open_positions)
    console.print(f"[bold magenta]{num_of_pos}[/bold magenta] open positions currently => {open_positions}")
    
    if num_of_pos > max_positions:
        console.print(
            f"[red]We have {num_of_pos} positions, which exceeds max_positions {max_positions} => closing...[/red]"
        )
        for coin in open_positions:
            kill_switch(coin, account)
    else:
        console.print(
            f"[green]We have {num_of_pos} positions <= max {max_positions}, no forced closes[/green]"
        )
    
    # Check specific symbol's position
    for pos in user_state["assetPositions"]:
        if (pos["position"]["coin"] == symbol) and float(pos["position"]["szi"]) != 0:
            positions.append(pos["position"])
            in_pos = True
            size = float(pos["position"]["szi"])
            pos_sym = pos["position"]["coin"]
            entry_px = float(pos["position"]["entryPx"])
            pnl_perc = float(pos["position"]["returnOnEquity"]) * 100
            console.print(f"[bold cyan]PnL %[/bold cyan]: {pnl_perc:.2f}% for {pos_sym}")
            break
    
    if size > 0:
        long_pos = True
    elif size < 0:
        long_pos = False
    
    return positions, in_pos, size, pos_sym, entry_px, pnl_perc, long_pos, num_of_pos


def close_all_positions(account):
    """
    Cancels all open orders and closes all open positions for the account.
    """
    console.print("[bold red]Close All Positions Triggered[/bold red]")
    
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    
    console.print(
        f"Account value => [bold green]{user_state['marginSummary']['accountValue']}[/bold green]"
    )
    
    # Cancel all orders
    cancel_all_orders(account)
    console.print("[bold yellow]All orders have been cancelled[/bold yellow]")
    
    open_positions = []
    for position in user_state["assetPositions"]:
        if float(position["position"]["szi"]) != 0:
            open_positions.append(position["position"]["coin"])
    
    for pos_symbol in open_positions:
        kill_switch(pos_symbol, account)
    
    console.print("[bold green]All positions have been closed[/bold green]")


# ---------------------------------------------------------------------
# NEW HELPER: Print a table of the "current situation" / status
# ---------------------------------------------------------------------
def print_current_status(symbol, account):
    """
    Prints a Rich table showing the current symbol's position side,
    size, entry, PnL %, and overall account value.
    """
    # Fetch position details
    positions, in_pos, size, pos_sym, entry_px, pnl_perc, is_long = get_position(symbol, account)
    val = acct_bal(account)

    # Build a small Rich table
    table = Table(title="Bot Current Status", box=box.ROUNDED)
    table.add_column("Symbol", justify="center", style="magenta")
    table.add_column("Pos Side", justify="center", style="cyan")
    table.add_column("Size", justify="right", style="green")
    table.add_column("Entry PX", justify="right", style="yellow")
    table.add_column("PnL %", justify="right", style="bold magenta")
    table.add_column("Account Val", justify="right", style="bold green")

    pos_side_str = "NONE"
    if in_pos:
        pos_side_str = "LONG" if is_long else "SHORT"

    table.add_row(
        f"{symbol}",
        pos_side_str,
        f"{abs(size):.4f}",
        f"{entry_px:.4f}",
        f"{pnl_perc:.2f}%",
        f"{val:.2f}"
    )

    console.print(table)


# ---------------------------
# Example of a main loop that uses print_current_status
# ---------------------------
def main_live_bot():
    """
    Example main live bot loop that runs every 4 hours.
    You would adapt your real strategy logic here, but at the end,
    we print the status table and sleep.
    """
    
    while True:
        console.rule("[bold green]Bot Cycle Start[/bold green]")
        
        # Example: do your strategy calls here...
        # run_ichimoku_bot(symbol, timeframe, account)
        # or whatever your logic function is, then:
        
        # Now print the status:
        print_current_status(symbol, account)
        
        # Sleep for 4 hours (4*3600 = 14400 seconds) or any timeframe you like
        console.print("[bold white on black]Sleeping for 4 hours...[/bold white on black]\n")
        time.sleep(4 * 3600)


if __name__ == "__main__":
    main_live_bot()
