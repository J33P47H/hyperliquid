###noice funcs
import dontshare as d
import noice_funcs_rich as n
from eth_account.signers.local import LocalAccount
import eth_account
from datetime import datetime, timedelta
import time

symbol = 'CRV'
timeframe = '4h'
coin = symbol 
secret_key = d.private_key
account: LocalAccount = eth_account.Account.from_key(secret_key)
try:
    while True:
        # print(n.acct_bal(account))
        # print(n.ask_bid(coin))
        # print(n.get_sz_px_decimals(coin))

        # leverage = 7
        # leverage, size = n.adjust_leverage_size_signal(symbol, leverage, account)
        # print(leverage, size)

        # is_buy = True
        # sz = 30
        # limit_px = 0.45
        # reduce_only = False
        # n.limit_order(coin, is_buy, sz, limit_px, reduce_only, account)

        # n.cancel_all_orders(account)



        # interval = '1m'
        # lookback_days = 30
        # data = n.get_ohlcv(symbol, interval, lookback_days)
        # # print(data)

        # df = n.process_data_to_df(data, write_to_csv=True)
        # print(df)

        interval = '30m'
        start_time = datetime.now() - timedelta(days=3)
        end_time = datetime.now()

        data = n.fetch_candle_snapshot(symbol, interval, start_time, end_time)
        # print(data)

        # n.get_position(symbol, account)
        # n.kill_switch(symbol, account)
        # max_loss = -5
        # target = 10
        # n.pnl_close(symbol, target, max_loss, account)
        # max_positions = 3
        # n.get_position_and_max_position(symbol, account, max_positions)

        # n.close_all_positions(account)
        print("Running...")
        time.sleep(10)

except KeyboardInterrupt:
    print("\nStopped by user")