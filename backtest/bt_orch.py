import pandas as pd
from backtesting import Backtest
import json
from datetime import datetime
from pathlib import Path
import importlib
import sys
import itertools
from typing import Dict, List, Any
import uuid

# === RICH IMPORTS ===
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import track

# Initialize a global console
console = Console(highlight=False, log_path=False)

# Add the strategies directory to Python path
STRATEGIES_DIR = Path("strategies")
DATA_DIR = Path("data/ohlcv")
RESULTS_DIR = Path("data/results")
LOG_DIR = Path("data/logs")
TRADE_LOG_DIR = LOG_DIR / "trades"

# Create directories
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)

def log_trades(trades_df: pd.DataFrame, stats: dict, strategy_name: str, symbol: str, run_id: str):
    """
    Log individual trade details to a CSV file with enhanced information.
    
    Parameters:
    -----------
    trades_df : pd.DataFrame
        DataFrame containing trade information
    stats : dict
        Statistics from the backtest run
    strategy_name : str
        Name of the strategy
    symbol : str
        Trading symbol
    run_id : str
        Unique identifier for the backtest run
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"trades_{strategy_name}_{symbol}_{timestamp}_{run_id[:8]}.csv"
    
    # Enhance trades DataFrame with additional information
    enhanced_trades = trades_df.copy()
    
    # Add basic trade information if not already present
    if 'Size' not in enhanced_trades.columns:
        enhanced_trades['Size'] = enhanced_trades['Size_Exit']
    
    # Calculate additional metrics for each trade
    enhanced_trades['Duration'] = enhanced_trades['ExitTime'] - enhanced_trades['EntryTime']
    enhanced_trades['Return_Pct'] = enhanced_trades['PnL'] / enhanced_trades['EntryPrice'] * 100
    enhanced_trades['Cumulative_Return_Pct'] = enhanced_trades['Return_Pct'].cumsum()
    enhanced_trades['Running_Max_Drawdown'] = enhanced_trades['Return_Pct'].cummin()
    
    # Add trade number and additional metrics
    enhanced_trades['Trade_Number'] = range(1, len(enhanced_trades) + 1)
    enhanced_trades['Cumulative_Trades'] = range(1, len(enhanced_trades) + 1)
    enhanced_trades['Win'] = enhanced_trades['PnL'] > 0
    enhanced_trades['Running_Win_Rate'] = enhanced_trades['Win'].expanding().mean() * 100
    
    # Calculate running statistics
    enhanced_trades['Running_Mean_Return'] = enhanced_trades['Return_Pct'].expanding().mean()
    enhanced_trades['Running_Std_Return'] = enhanced_trades['Return_Pct'].expanding().std()
    enhanced_trades['Running_Sharpe'] = (enhanced_trades['Running_Mean_Return'] / 
                                       enhanced_trades['Running_Std_Return'])
    
    # Add trade direction
    enhanced_trades['Direction'] = 'Long'
    enhanced_trades.loc[enhanced_trades['Size'] < 0, 'Direction'] = 'Short'
    
    # Add trade result classification
    enhanced_trades['Result'] = 'Win'
    enhanced_trades.loc[enhanced_trades['PnL'] < 0, 'Result'] = 'Loss'
    enhanced_trades.loc[enhanced_trades['PnL'] == 0, 'Result'] = 'Breakeven'
    
    # Add strategy and run information
    enhanced_trades['Strategy'] = strategy_name
    enhanced_trades['Symbol'] = symbol
    enhanced_trades['Run_ID'] = run_id
    
    # Reorder columns for better readability
    columns_order = [
        'Trade_Number',
        'Strategy',
        'Symbol',
        'Direction',
        'EntryTime',
        'ExitTime',
        'Duration',
        'EntryPrice',
        'ExitPrice',
        'Size',
        'PnL',
        'Return_Pct',
        'Result',
        'Cumulative_Return_Pct',
        'Running_Win_Rate',
        'Running_Mean_Return',
        'Running_Std_Return',
        'Running_Sharpe',
        'Running_Max_Drawdown',
        'Run_ID'
    ]
    
    # Add any remaining columns from the original DataFrame
    remaining_cols = [col for col in enhanced_trades.columns if col not in columns_order]
    columns_order.extend(remaining_cols)
    
    # Reorder and save
    enhanced_trades = enhanced_trades[columns_order]
    enhanced_trades.to_csv(TRADE_LOG_DIR / filename, index=True)
    
    # Also save trade summary
    summary_filename = f"trade_summary_{strategy_name}_{symbol}_{timestamp}_{run_id[:8]}.csv"
    trade_summary = pd.DataFrame({
        'Metric': [
            'Total_Trades',
            'Winning_Trades',
            'Losing_Trades',
            'Win_Rate',
            'Best_Trade_Return',
            'Worst_Trade_Return',
            'Avg_Trade_Return',
            'Avg_Winner_Return',
            'Avg_Loser_Return',
            'Largest_Winner',
            'Largest_Loser',
            'Avg_Trade_Duration',
            'Max_Consecutive_Winners',
            'Max_Consecutive_Losers',
            'Profit_Factor',
            'Recovery_Factor',
            'Risk_Reward_Ratio',
            'Expectancy',
            'Standard_Deviation',
            'Sharpe_Ratio',
            'Sortino_Ratio'
        ],
        'Value': [
            len(enhanced_trades),
            sum(enhanced_trades['Win']),
            sum(~enhanced_trades['Win']),
            stats['Win Rate [%]'],
            stats['Best Trade [%]'],
            stats['Worst Trade [%]'],
            stats['Avg. Trade [%]'],
            enhanced_trades.loc[enhanced_trades['Win'], 'Return_Pct'].mean(),
            enhanced_trades.loc[~enhanced_trades['Win'], 'Return_Pct'].mean(),
            enhanced_trades['PnL'].max(),
            enhanced_trades['PnL'].min(),
            stats['Avg. Trade Duration'],
            stats.get('Max Consecutive Winners', 'N/A'),
            stats.get('Max Consecutive Losers', 'N/A'),
            stats['Profit Factor'],
            stats.get('Recovery Factor', 'N/A'),
            abs(enhanced_trades.loc[enhanced_trades['Win'], 'Return_Pct'].mean() / 
                enhanced_trades.loc[~enhanced_trades['Win'], 'Return_Pct'].mean()),
            stats['Expectancy [%]'],
            enhanced_trades['Return_Pct'].std(),
            stats['Sharpe Ratio'],
            stats['Sortino Ratio']
        ]
    })
    
    trade_summary.to_csv(TRADE_LOG_DIR / summary_filename, index=False)
    
    return filename

def log_backtest_run(run_data: dict, log_file: str = "backtest_detailed_log.csv"):
    """
    Log detailed information about each backtest run to a CSV file.
    
    Parameters:
    -----------
    run_data : dict
        Dictionary containing run information
    log_file : str
        Name of the log file
    """
    log_path = LOG_DIR / log_file
    run_df = pd.DataFrame([run_data])
    
    if log_path.exists():
        run_df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        run_df.to_csv(log_path, index=False)

# Add strategies directory to Python path if it exists
if STRATEGIES_DIR.exists():
    sys.path.append(str(STRATEGIES_DIR.absolute()))


def load_data(csv_path):
    """
    Loads intraday trading data from CSV with proper timestamp parsing.
    Returns a DataFrame suitable for backtesting.py (with Open, High, Low, Close, Volume).
    """
    console.print(Panel.fit(f"[bold cyan]📂 Loading data from:[/bold cyan] [white]{csv_path}[/white]", style="cyan"))
    df = pd.read_csv(csv_path)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    df = df.sort_values('timestamp')
    df.set_index('timestamp', inplace=True)
    
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    console.log(f"✅ DataFrame shape: [bold yellow]{df.shape}[/bold yellow]")
    return df


def generate_parameter_combinations(param_config, max_combinations: int = 100) -> List[Dict[str, Any]]:
    """
    Generate all possible parameter combinations from the strategy class's param_config.
    Ensures the default combination is always included, and limits to max_combinations if too large.
    """
    console.log("[bold cyan]🔧 Generating parameter combinations...[/bold cyan]")
    param_ranges = {}
    default_params = {}
    
    for param_name, param_info in param_config.items():
        if isinstance(param_info, dict):
            if 'range' in param_info:
                param_ranges[param_name] = param_info['range']
            if 'default' in param_info:
                default_params[param_name] = param_info['default']
    
    if not param_ranges:
        console.log("[bold yellow]No param ranges found; returning default params only.[/bold yellow]")
        return [default_params] if default_params else []
    
    from itertools import product
    keys = list(param_ranges.keys())
    values = [param_ranges[key] for key in keys]
    all_combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    
    if default_params not in all_combinations:
        all_combinations.append(default_params)
    
    if len(all_combinations) > max_combinations:
        import random
        random.seed(42)
        sampled = random.sample(all_combinations, max_combinations - 1)
        if default_params not in sampled:
            sampled.append(default_params)
        all_combinations = sampled
        console.log(
            f"[bold yellow]⚠️ Truncated parameter combinations to {max_combinations}[/bold yellow]"
        )
    
    console.log(
        f"[green]🎉 Total parameter combos generated:[/green] {len(all_combinations)}"
    )
    return all_combinations


def run_strategy_backtest(df, symbol, strategy_class, strategy_params=None, cash=10_000, commission=0.001, save_trades=True, is_optimization=False):
    """
    Run a backtest for a single strategy with optional parameters, returning (stats, backtest_instance).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data
    symbol : str
        Trading symbol
    strategy_class : class
        Strategy class to test
    strategy_params : dict, optional
        Strategy parameters
    cash : float, optional
        Initial cash
    commission : float, optional
        Commission rate
    save_trades : bool, optional
        Whether to save trade logs (default: True)
    is_optimization : bool, optional
        Whether this run is part of optimization (suppresses output)
    """
    run_id = str(uuid.uuid4())
    run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    log_data = {
        'run_id': run_id,
        'timestamp': run_timestamp,
        'symbol': symbol,
        'strategy': strategy_class.__name__,
        'parameters': str(strategy_params),
        'initial_cash': cash,
        'commission': commission,
        'status': 'started',
        'error_message': '',
        'duration': '',
        'n_trades': 0,
        'final_equity': 0,
        'return_pct': 0,
        'cagr_pct': 0,
        'buy_hold_return_pct': 0,
        'sharpe_ratio': 0,
        'max_drawdown': 0,
        'win_rate': 0,
        'trades_file': ''
    }
    
    try:
        start_time = datetime.now()
        
        # Copy data to avoid modifications
        df = df.copy()
        
        # Calculate Buy & Hold return
        buy_hold_return = calculate_buy_and_hold(df)
        
        # Create strategy instance with symbol
        strategy_instance = strategy_class
        if hasattr(strategy_class, 'symbol'):
            strategy_instance.symbol = symbol
        
        bt = Backtest(df, strategy_instance, cash=cash, commission=commission)
        
        if not is_optimization:
            console.log(
                f"🏁[bold yellow]{symbol}  Running [bold white]{strategy_class.__name__}[/bold white] strategy "
                f"with [bold cyan]{strategy_params}[/bold cyan]..."
            )
        
        if strategy_params and isinstance(strategy_params, dict):
            stats = bt.run(**strategy_params)
        else:
            stats = bt.run()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Log trades if they exist and save_trades is True
        trades_file = ''
        if save_trades and hasattr(stats, '_trades') and not stats._trades.empty:
            trades_file = log_trades(stats._trades, stats, strategy_class.__name__, symbol, run_id)
        
        # Update log data with results
        log_data.update({
            'status': 'completed',
            'duration': duration,
            'n_trades': stats['# Trades'],
            'final_equity': stats['Equity Final [$]'],
            'return_pct': stats['Return [%]'],
            'cagr_pct': stats['CAGR [%]'],
            'buy_hold_return_pct': buy_hold_return,
            'sharpe_ratio': stats['Sharpe Ratio'],
            'max_drawdown': stats['Max. Drawdown [%]'],
            'win_rate': stats['Win Rate [%]'],
            'trades_file': trades_file
        })
        
        if not is_optimization:
            console.log(
                f"✅ [bold green]Backtest complete[/bold green]. "
                f"# Trades: {stats['# Trades']}, Final Equity: {stats['Equity Final [$]']:.2f}"
            )
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        log_data.update({
            'status': 'failed',
            'error_message': str(e),
            'duration': duration
        })
        raise e
    
    finally:
        # Log the run regardless of success or failure
        log_backtest_run(log_data)
    
    return stats, bt


def calculate_buy_and_hold(df):
    """
    Calculate a simple Buy & Hold return from the first to the last 'Close' in the dataset.
    """
    first_price = df['Close'].iloc[0]
    last_price = df['Close'].iloc[-1]
    return ((last_price - first_price) / first_price) * 100


def save_results(stats, strategy_name, symbol, timeframe, params=None, output_dir=RESULTS_DIR):
    """
    Save backtest results to a CSV (backtest_summary.csv).
    Creates or updates the CSV file by adding a new column for each run.
    """
    from rich.table import Table
    from rich.panel import Panel

    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        num_trades = stats['# Trades']
        commission_rate = 0.001  
        avg_trade_size = stats['Equity Final [$]'] / (num_trades if num_trades > 0 else 1)
        total_commissions = num_trades * avg_trade_size * commission_rate if num_trades > 0 else 0
        
        results_dict = {
            'Start': stats['Start'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(stats['Start'], 'strftime') else stats['Start'],
            'End': stats['End'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(stats['End'], 'strftime') else stats['End'],
            'Duration': stats['Duration'],
            'Exposure Time [%]': stats['Exposure Time [%]'],
            'Equity Final [$]': stats['Equity Final [$]'],
            'Equity Peak [$]': stats['Equity Peak [$]'],
            'Commissions [$]': total_commissions,
            'Return [%]': stats['Return [%]'],
            'Buy & Hold Return [%]': stats['Buy & Hold Return [%]'],
            'Return (Ann.) [%]': stats['Return (Ann.) [%]'],
            'Volatility (Ann.) [%]': stats['Volatility (Ann.) [%]'],
            'CAGR [%]': stats['CAGR [%]'],
            'Sharpe Ratio': stats['Sharpe Ratio'],
            'Sortino Ratio': stats['Sortino Ratio'],
            'Calmar Ratio': stats['Calmar Ratio'],
            'Max. Drawdown [%]': stats['Max. Drawdown [%]'],
            'Avg. Drawdown [%]': stats['Avg. Drawdown [%]'],
            'Max. Drawdown Duration': stats['Max. Drawdown Duration'],
            'Avg. Drawdown Duration': stats['Avg. Drawdown Duration'],
            '# Trades': stats['# Trades'],
            'Win Rate [%]': stats['Win Rate [%]'],
            'Best Trade [%]': stats['Best Trade [%]'],
            'Worst Trade [%]': stats['Worst Trade [%]'],
            'Avg. Trade [%]': stats['Avg. Trade [%]'],
            'Max. Trade Duration': stats['Max. Trade Duration'],
            'Avg. Trade Duration': stats['Avg. Trade Duration'],
            'Profit Factor': stats['Profit Factor'],
            'Expectancy [%]': stats['Expectancy [%]'],
            'SQN': stats['SQN'],
            'Kelly Criterion': stats['Kelly Criterion']
        }
        
        # Attach params if any
        if params:
            for param_name, param_value in params.items():
                results_dict[f'param_{param_name}'] = param_value
        
        results_df = pd.DataFrame.from_dict(results_dict, orient='index')
        filename = output_dir / "backtest_summary.csv"
        
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_column_name = f"{strategy_name}_{symbol}_{timeframe}_{now_str}"
        results_df.columns = [new_column_name]
        
        # Show a small summary table in the console
        table = Table(title=f'Backtest Results Summary 🌟for {symbol}', box=box.SIMPLE_HEAVY)
        table.add_column("Metric", justify="left", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        # Let's pick a few metrics to display in the console
        display_keys = ["Start", "End", "Duration",
                        "Exposure Time [%]", "Equity Final [$]",
                        "Equity Peak [$]", "Commissions [$]",
                        "Return [%]", "Buy & Hold Return [%]",
                        "Return (Ann.) [%]", "Volatility (Ann.) [%]",
                        "CAGR [%]", "Sharpe Ratio",
                        "Sortino Ratio", "Calmar Ratio",
                        "Max. Drawdown [%]", "Avg. Drawdown [%]",
                        "Max. Drawdown Duration", "Avg. Drawdown Duration",
                        "# Trades", "Win Rate [%]",
                        "Best Trade [%]", "Worst Trade [%]",
                        "Avg. Trade [%]", "Max. Trade Duration",
                        "Avg. Trade Duration", "Profit Factor",
                        "Expectancy [%]","SQN", "Kelly Criterion"]
        for k in display_keys:
            val = results_dict.get(k, "N/A")
            table.add_row(k, str(val))
        
        console.print(Panel(table, title=f"[bold magenta]{strategy_name} - {symbol} ({timeframe})[/bold magenta]"))
        
        console.print(f"[yellow]💾 Saving results to:[/yellow] {filename}")
        
        # Handle existing file
        if filename.exists():
            try:
                existing_df = pd.read_csv(filename, index_col=0)
                
                # Remove older runs of the same strategy/symbol/timeframe
                existing_cols = existing_df.columns
                strategy_cols = [c for c in existing_cols if c.startswith(f"{strategy_name}_{symbol}_{timeframe}")]
                if strategy_cols:
                    existing_df = existing_df.drop(columns=strategy_cols)
                
                combined_df = pd.concat([existing_df, results_df], axis=1)
                combined_df.to_csv(filename)
                console.print("🔄 [green]Updated existing file[/green]")
            except Exception as e:
                console.print(f"[bold red]Error appending to file:[/bold red] {str(e)}")
                results_df.to_csv(filename)
                console.print("[green]Created new file[/green]")
        else:
            results_df.to_csv(filename)
            console.print("[green]Created new file[/green]")
        
        return results_df
    except Exception as e:
        console.print(f"[bold red]❌ Error saving results:[/bold red] {str(e)}")
        return None


def load_strategy(strategy_name):
    """
    Dynamically load strategy class from a file named bt_<lowercase name>_s.py,
    expecting a class <Name>Strategy inside.
    """
    try:
        module_name = f"bt_{strategy_name.lower()}_s"
        module = importlib.import_module(module_name)
        strategy_class = getattr(module, f"{strategy_name}Strategy")
        console.log(f"[bold green]🎯 Loaded strategy class:[/bold green] {strategy_class.__name__}")
        return strategy_class
    except Exception as e:
        console.print(f"[bold red]❌ Error loading strategy {strategy_name}:[/bold red] {str(e)}")
        return None


def load_config(config_file='backtest_config.json'):
    """
    Load configuration from a JSON file. 
    Adjust data source paths to match the DATA_DIR structure.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            
            for key, path in config['data_sources'].items():
                config['data_sources'][key] = str(DATA_DIR / Path(path).name)
                
        console.log(f"⚙️ Loaded config from [bold]{config_file}[/bold]")
        return config
    except FileNotFoundError:
        console.print("[bold red]⚠️ Config file not found.[/bold red] Using default configuration.")
        return None


def optimize_strategy(df, symbol, strategy_name, strategy_class, strategy_config, backtest_settings):
    """
    Run parameter optimization for a strategy with different param combos from param_config.
    Returns (best_stats, best_params, best_metric_value).
    """
    optimization_id = str(uuid.uuid4())
    optimization_start = datetime.now()
    
    console.print(f"\n[bold blue]🔎 Optimizing {strategy_name} strategy...[/bold blue]")
    try:
        param_config = getattr(strategy_class, 'param_config', {})
        if not param_config:
            raise Exception(f"No parameter configuration found for {strategy_name}")
        
        param_combinations = generate_parameter_combinations(
            param_config,
            backtest_settings.get('max_combinations', 100)
        )
        
        best_stats = None
        best_params = None
        best_metric_value = float('-inf')
        best_trades = None
        optimization_metric = backtest_settings.get('optimization_metric', 'Sharpe Ratio')
        
        total_combos = len(param_combinations)
        console.print(f"[bold yellow]Testing {total_combos} parameter combinations...[/bold yellow]")
        
        # Calculate Buy & Hold return once
        buy_hold_return = calculate_buy_and_hold(df)
        
        # Log optimization start
        log_data = {
            'run_id': optimization_id,
            'timestamp': optimization_start.strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'strategy': strategy_name,
            'parameters': 'optimization_start',
            'total_combinations': total_combos,
            'optimization_metric': optimization_metric,
            'status': 'optimization_started',
            'best_metric_value': None,
            'cagr_pct': None,
            'buy_hold_return_pct': buy_hold_return
        }
        log_backtest_run(log_data, "optimization_log.csv")
        
        for combo in track(param_combinations, description="Optimizing... 🚀"):
            try:
                # Run backtest without saving trades during optimization
                stats, bt_instance = run_strategy_backtest(
                    df,
                    symbol,
                    strategy_class,
                    combo,
                    cash=backtest_settings['initial_capital'],
                    commission=backtest_settings['commission'],
                    save_trades=False,  # Don't save trades during optimization
                    is_optimization=True
                )
                
                if stats['# Trades'] == 0:
                    continue
                
                current_metric = stats[optimization_metric]
                if current_metric is None or (isinstance(current_metric, float) and pd.isna(current_metric)):
                    continue
                
                if current_metric > best_metric_value:
                    best_metric_value = current_metric
                    best_stats = stats
                    best_params = combo
                    # Store trades from the best run
                    if hasattr(stats, '_trades'):
                        best_trades = stats._trades.copy()
                    
                    # Log new best result
                    log_data = {
                        'run_id': optimization_id,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'symbol': symbol,
                        'strategy': strategy_name,
                        'parameters': str(combo),
                        'status': 'new_best',
                        'best_metric_value': best_metric_value,
                        'trades': stats['# Trades'],
                        'win_rate': stats['Win Rate [%]'],
                        'return_pct': stats['Return [%]']
                    }
                    log_backtest_run(log_data, "optimization_log.csv")
                    
                    # Print only when new best is found
                    console.print(f"\n🏆 [bright_white]New best[/bright_white] {optimization_metric}: [green]{best_metric_value:.4f}[/green]")
                    console.print(f"Parameters: [cyan]{best_params}[/cyan]")
                    console.print(f"Trades: {stats['# Trades']}, Win Rate: {stats['Win Rate [%]']:.1f}%, Return: {stats['Return [%]']:.1f}%\n")
                    
            except Exception as e:
                # Log failed parameter combination but don't print unless it's a new type of error
                log_data = {
                    'run_id': optimization_id,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': symbol,
                    'strategy': strategy_name,
                    'parameters': str(combo),
                    'status': 'failed',
                    'error_message': str(e)
                }
                log_backtest_run(log_data, "optimization_log.csv")
                continue
        
        if best_stats is None:
            raise Exception("No valid parameter combinations found")
        
        # Save trades only for the best run
        trades_file = ''
        if best_trades is not None:
            trades_file = log_trades(best_trades, best_stats, strategy_name, symbol, optimization_id)
        
        # Log optimization completion
        log_data = {
            'run_id': optimization_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'strategy': strategy_name,
            'parameters': str(best_params),
            'status': 'optimization_completed',
            'best_metric_value': best_metric_value,
            'duration': (datetime.now() - optimization_start).total_seconds(),
            'trades_file': trades_file
        }
        log_backtest_run(log_data, "optimization_log.csv")
        
        console.print(f"\n[bold green]✅ Optimization completed![/bold green]")
        console.print(f"Best {optimization_metric}: [cyan]{best_metric_value:.4f}[/cyan]")
        
        return best_stats, best_params, best_metric_value
        
    except Exception as e:
        # Log optimization failure
        log_data = {
            'run_id': optimization_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'strategy': strategy_name,
            'status': 'optimization_failed',
            'error_message': str(e),
            'duration': (datetime.now() - optimization_start).total_seconds()
        }
        log_backtest_run(log_data, "optimization_log.csv")
        
        console.print(f"[bold red]❌ Error during optimization:[/bold red] {str(e)}")
        return None, None, float('-inf')


def main():
    """
    Main entry point: 
    1) Load config, 
    2) Create directories, 
    3) Run each strategy (and optionally optimize), 
    4) Save results.
    """
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    config = load_config()
    if not config:
        console.print("[bold red]No configuration loaded. Exiting.[/bold red]")
        return
    
    # Filter for enabled strategies
    enabled_strategies = {
        name: info for name, info in config['strategies'].items()
        if info.get('enabled', True)
    }
    
    console.rule("[bold magenta]🔬 Starting Backtest Process[/bold magenta]")
    console.print(
        f"🔹 Enabled strategies: [bold]{list(enabled_strategies.keys())}[/bold]"
    )
    
    for symbol_tf, csv_path in config['data_sources'].items():
        # symbol_tf might be something like "BTC_4h"
        if "_" in symbol_tf:
            symbol, timeframe = symbol_tf.split('_', 1)
        else:
            # fallback if no underscore
            symbol, timeframe = symbol_tf, "unknownTF"
        
        console.rule(f"[bold cyan]📊 Processing {symbol} ({timeframe})[/bold cyan]")
        
        try:
            df = load_data(csv_path)
            
            for strategy_name, strategy_info in enabled_strategies.items():
                console.print(
                    f"\n[bold yellow]🛠 Running {strategy_name} strategy[/bold yellow] "
                    f"on [blue]{symbol_tf}[/blue]..."
                )
                
                strategy_class = load_strategy(strategy_name)
                if not strategy_class:
                    continue
                
                if strategy_info.get('optimize', False):
                    best_stats, best_params, best_metric = optimize_strategy(
                        df,
                        symbol,
                        strategy_name,
                        strategy_class,
                        strategy_info,
                        config['backtest_settings']
                    )
                    
                    if best_stats is not None and best_metric != float('-inf'):
                        saved_df = save_results(
                            best_stats,
                            strategy_name,
                            symbol,
                            timeframe,
                            params=best_params
                        )
                        
                        if saved_df is not None:
                            console.print(f"[green]🎉 {strategy_name} optimization results saved successfully[/green]")
                    else:
                        console.print(f"[bold red]Warning:[/bold red] Optimization failed for {strategy_name}")
                else:
                    # Single run with default params
                    param_config = getattr(strategy_class, 'param_config', {})
                    default_params = {
                        p: cfg['default']
                        for p, cfg in param_config.items()
                        if isinstance(cfg, dict) and 'default' in cfg
                    }
                    
                    stats, _ = run_strategy_backtest(
                        df,
                        symbol,
                        strategy_class,
                        default_params,
                        cash=config['backtest_settings']['initial_capital'],
                        commission=config['backtest_settings']['commission']
                    )
                    
                    saved_df = save_results(
                        stats,
                        strategy_name,
                        symbol,
                        timeframe,
                        params=default_params
                    )
                    if saved_df is not None:
                        console.print(f"[green]🎉 {strategy_name} backtest results saved successfully[/green]")
                
                console.print(f"[white]✨ {strategy_name} completed successfully[/white]")
                
        except Exception as e:
            console.print(f"[bold red]❌ Error processing {symbol}:[/bold red] {str(e)}")
            continue
    
    console.rule("[bold green]🏁 All Done! 🏁[/bold green]")


if __name__ == "__main__":
    main()
