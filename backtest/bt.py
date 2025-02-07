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
import numpy as np

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

# Add strategies directory to Python path if it exists
if STRATEGIES_DIR.exists():
    sys.path.append(str(STRATEGIES_DIR.absolute()))

# Create directories
LOG_DIR.mkdir(parents=True, exist_ok=True)

def generate_parameter_combinations(param_config, max_combinations: int = 100) -> List[Dict[str, Any]]:
    """
    Generate parameter combinations from start/end/step configuration.
    Ensures the default combination is always included.
    
    Parameters:
    -----------
    param_config : dict
        Dictionary containing parameter configurations with start/end/step values
    max_combinations : int
        Maximum number of combinations to generate
    
    Returns:
    --------
    list
        List of parameter combinations
    """
    console.log("[bold cyan]🔧 Generating parameter combinations...[/bold cyan]")
    param_ranges = {}
    default_params = {}
    
    for param_name, param_info in param_config.items():
        if isinstance(param_info, dict):
            if all(key in param_info for key in ['start', 'end', 'step']):
                # Generate range using np.arange for more precise step control
                param_ranges[param_name] = list(np.arange(
                    param_info['start'],
                    param_info['end'] + param_info['step'],  # Include end value
                    param_info['step']
                ))
            if 'default' in param_info:
                default_params[param_name] = param_info['default']
    
    if not param_ranges:
        console.log("[bold yellow]No param ranges found; returning default params only.[/bold yellow]")
        return [default_params] if default_params else []
    
    # Generate all combinations
    keys = list(param_ranges.keys())
    values = [param_ranges[key] for key in keys]
    all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    # Ensure default combination is included
    if default_params and default_params not in all_combinations:
        all_combinations.append(default_params)
    
    # Sample if too many combinations
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

def log_trades(trades_df: pd.DataFrame, stats: dict, strategy_name: str, symbol: str, run_id: str, timeframe: str):
    """
    Log top 10 best and worst trades to a CSV file.
    """
    timestamp = datetime.now().strftime('%Y%m%d')
    original_filename = f"{strategy_name}_{symbol}_{timeframe}_{timestamp}"
    filename = "trades.csv"
    
    # Enhance trades DataFrame with additional information
    enhanced_trades = trades_df.copy()
    
    # Add basic trade information if not already present
    if 'Size' not in enhanced_trades.columns:
        enhanced_trades['Size'] = enhanced_trades['Size_Exit']
    
    # Calculate additional metrics for each trade
    enhanced_trades['Duration'] = enhanced_trades['ExitTime'] - enhanced_trades['EntryTime']
    enhanced_trades['Return_Pct'] = enhanced_trades['PnL'] / enhanced_trades['EntryPrice'] * 100
    enhanced_trades['Trade_Number'] = range(1, len(enhanced_trades) + 1)
    enhanced_trades['Direction'] = 'Long'
    enhanced_trades.loc[enhanced_trades['Size'] < 0, 'Direction'] = 'Short'
    
    # Add strategy and run information
    enhanced_trades['Strategy'] = strategy_name
    enhanced_trades['Symbol'] = symbol
    enhanced_trades['Timeframe'] = timeframe
    enhanced_trades['Run_ID'] = run_id
    enhanced_trades['Filename'] = original_filename
    
    # Select columns to keep
    columns_order = [
        'Trade_Number',
        'Strategy',
        'Symbol',
        'Timeframe',
        'Direction',
        'EntryTime',
        'ExitTime',
        'Duration',
        'EntryPrice',
        'ExitPrice',
        'Size',
        'PnL',
        'Return_Pct',
        'Run_ID',
        'Filename'
    ]
    
    # Keep only selected columns
    enhanced_trades = enhanced_trades[columns_order]
    
    # Get top 10 best and worst trades based on Return_Pct
    best_trades = enhanced_trades.nlargest(10, 'Return_Pct')
    worst_trades = enhanced_trades.nsmallest(10, 'Return_Pct')
    
    # Combine and prepare to save
    selected_trades = pd.concat([best_trades, worst_trades])
    
    # Handle existing file
    filepath = LOG_DIR / filename
    if filepath.exists():
        existing_df = pd.read_csv(filepath)
        # Remove older trades from the same run if they exist
        existing_df = existing_df[existing_df['Filename'] != original_filename]
        selected_trades = pd.concat([existing_df, selected_trades])
    
    selected_trades.to_csv(filepath, index=False)
    return filename

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

def run_strategy_backtest(df, symbol, strategy_class, strategy_params=None, cash=10_000, commission=0.001, save_trades=True, is_optimization=False, timeframe=''):
    """
    Run a backtest for a single strategy with optional parameters.
    """
    try:
        # Copy data to avoid modifications
        df = df.copy()
        
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
        
        # Log trades if they exist and save_trades is True
        if save_trades and hasattr(stats, '_trades') and not stats._trades.empty:
            run_id = str(uuid.uuid4())
            trades_file = log_trades(stats._trades, stats, strategy_class.__name__, symbol, run_id, timeframe)
        
        if not is_optimization:
            console.log(
                f"✅ [bold green]Backtest complete[/bold green]. "
                f"# Trades: {stats['# Trades']}, Final Equity: {stats['Equity Final [$]']:.2f}"
            )
        
        return stats, bt
        
    except Exception as e:
        console.print(f"[bold red]❌ Error during backtest:[/bold red] {str(e)}")
        raise e

def calculate_buy_and_hold(df):
    """
    Calculate a simple Buy & Hold return from the first to the last 'Close' in the dataset.
    """
    first_price = df['Close'].iloc[0]
    last_price = df['Close'].iloc[-1]
    return ((last_price - first_price) / first_price) * 100

def save_results(stats, strategy_name, symbol, timeframe, params=None, output_dir=LOG_DIR):
    """
    Save backtest results to a CSV (backtest_summary.csv).
    Creates or updates the CSV file with transposed data.
    """
    from rich.table import Table
    from rich.panel import Panel

    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create index (row names) in the desired order
        index_order = [
            'Start',
            'End',
            'Duration',
            'Exposure Time [%]',
            'Equity Final [$]',
            'Equity Peak [$]',
            'Commissions [$]',
            'Return [%]',
            'Buy & Hold Return [%]',
            'Return (Ann.) [%]',
            'Volatility (Ann.) [%]',
            'CAGR [%]',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Calmar Ratio',
            'Max. Drawdown [%]',
            'Avg. Drawdown [%]',
            'Max. Drawdown Duration',
            'Avg. Drawdown Duration',
            '# Trades',
            'Win Rate [%]',
            'Best Trade [%]',
            'Worst Trade [%]',
            'Avg. Trade [%]',
            'Max. Trade Duration',
            'Avg. Trade Duration',
            'Profit Factor',
            'Expectancy [%]',
            'SQN',
            'Kelly Criterion'
        ]
        
        # Create results dictionary with ordered data
        results_dict = {}
        for key in index_order:
            value = stats[key]
            if isinstance(value, pd.Timestamp):
                value = value.strftime('%Y-%m-%d %H:%M:%S')
            results_dict[key] = value
        
        # Create DataFrame with single row
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{strategy_name}_{symbol}_{timeframe}_{now_str}"
        results_df = pd.DataFrame([results_dict], index=[run_name])
        
        filename = output_dir / "backtest_summary.csv"
        
        # Show a small summary table in the console
        table = Table(title=f'Backtest Results Summary 🌟for {symbol}', box=box.SIMPLE_HEAVY)
        table.add_column("Metric", justify="left", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        for k in index_order:
            val = results_dict.get(k, "N/A")
            table.add_row(k, str(val))
        
        console.print(Panel(table, title=f"[bold magenta]{strategy_name} - {symbol} ({timeframe})[/bold magenta]"))
        
        console.print(f"[yellow]💾 Saving results to:[/yellow] {filename}")
        
        # Handle existing file
        if filename.exists():
            try:
                existing_df = pd.read_csv(filename, index_col=0)
                
                # Remove older runs of the same strategy/symbol/timeframe
                existing_indices = existing_df.index
                strategy_indices = [i for i in existing_indices if i.startswith(f"{strategy_name}_{symbol}_{timeframe}")]
                if strategy_indices:
                    existing_df = existing_df.drop(index=strategy_indices)
                
                # Combine with new results
                combined_df = pd.concat([existing_df, results_df])
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
    Dynamically load strategy class from a file named bt_<lowercase name>_s.py.
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

def log_optimization_results(optimization_results: list, strategy_name: str, symbol: str, optimization_id: str, timeframe: str):
    """
    Log the best 5 optimization runs to a CSV file.
    """
    timestamp = datetime.now().strftime('%Y%m%d')
    original_filename = f"{strategy_name}_{symbol}_{timeframe}_{timestamp}"
    filename = "optimization.csv"
    
    try:
        # Create DataFrame from optimization results
        results_df = pd.DataFrame(optimization_results)
        
        # Sort by Metric_Value and get top 5
        results_df = results_df.nlargest(5, 'Metric_Value')
        
        # Add run information
        results_df['Strategy'] = strategy_name
        results_df['Symbol'] = symbol
        results_df['Timeframe'] = timeframe
        results_df['Run_ID'] = optimization_id
        results_df['Timestamp'] = timestamp
        results_df['Filename'] = original_filename
        
        # Reorder columns
        columns_order = [
            'Run_Number',
            'Strategy',
            'Symbol',
            'Timeframe',
            'Parameters',
            'Metric_Name',
            'Metric_Value',
            'Total_Trades',
            'Win_Rate',
            'Return_Pct',
            'Max_Drawdown',
            'Sharpe_Ratio',
            'Timestamp',
            'Run_ID',
            'Filename'
        ]
        
        # Ensure all columns exist
        for col in columns_order:
            if col not in results_df.columns:
                results_df[col] = None
        
        results_df = results_df[columns_order]
        
        # Handle existing file
        filepath = LOG_DIR / filename
        if filepath.exists():
            existing_df = pd.read_csv(filepath)
            # Remove older results from the same run if they exist
            existing_df = existing_df[existing_df['Filename'] != original_filename]
            results_df = pd.concat([existing_df, results_df])
        
        results_df.to_csv(filepath, index=False)
        return filename
    except Exception as e:
        console.print(f"[bold red]Error saving optimization results:[/bold red] {str(e)}")
        return None

def optimize_strategy(df, symbol, strategy_name, strategy_class, strategy_config, backtest_settings, timeframe=''):
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
        optimization_metric = backtest_settings.get('optimization_metric', 'Equity Final [$]')
        
        # Store all successful runs for logging
        optimization_results = []
        run_counter = 0
        
        for combo in track(param_combinations, description="Optimizing... 🚀"):
            try:
                stats, bt_instance = run_strategy_backtest(
                    df,
                    symbol,
                    strategy_class,
                    combo,
                    cash=backtest_settings['initial_capital'],
                    commission=backtest_settings['commission'],
                    save_trades=False,
                    is_optimization=True,
                    timeframe=timeframe
                )
                
                if stats['# Trades'] == 0:
                    continue
                
                current_metric = stats[optimization_metric]
                if current_metric is None or (isinstance(current_metric, float) and pd.isna(current_metric)):
                    continue
                
                run_counter += 1
                
                # Store run results
                run_result = {
                    'Run_Number': run_counter,
                    'Parameters': str(combo),
                    'Metric_Name': optimization_metric,
                    'Metric_Value': current_metric,
                    'Total_Trades': stats['# Trades'],
                    'Win_Rate': stats['Win Rate [%]'],
                    'Return_Pct': stats['Return [%]'],
                    'Max_Drawdown': stats['Max. Drawdown [%]'],
                    'Sharpe_Ratio': stats['Sharpe Ratio']
                }
                optimization_results.append(run_result)
                
                if current_metric > best_metric_value:
                    best_metric_value = current_metric
                    best_stats = stats
                    best_params = combo
                    if hasattr(stats, '_trades'):
                        best_trades = stats._trades.copy()
                    
                    # Print only when new best is found
                    console.print(f"\n🏆 [bright_white]New best[/bright_white] {optimization_metric}: [green]{best_metric_value:.4f}[/green]")
                    console.print(f"Parameters: [cyan]{best_params}[/cyan]")
                    console.print(f"Trades: {stats['# Trades']}, Win Rate: {stats['Win Rate [%]']:.1f}%, Return: {stats['Return [%]']:.1f}%\n")
                    
            except Exception as e:
                console.print(f"[bold yellow]Warning:[/bold yellow] Failed run with parameters {combo}: {str(e)}")
                continue
        
        if best_stats is None:
            raise Exception("No valid parameter combinations found")
        
        # Save trades only for the best run
        if best_trades is not None:
            trades_file = log_trades(best_trades, best_stats, strategy_name, symbol, optimization_id, timeframe)
        
        # Log optimization results
        if optimization_results:
            optimization_file = log_optimization_results(
                optimization_results,
                strategy_name,
                symbol,
                optimization_id,
                timeframe
            )
            
            if optimization_file:
                console.print(f"[green]Optimization results saved to:[/green] {optimization_file}")
        
        console.print(f"\n[bold green]✅ Optimization completed![/bold green]")
        console.print(f"Best {optimization_metric}: [cyan]{best_metric_value:.4f}[/cyan]")
        
        return best_stats, best_params, best_metric_value
        
    except Exception as e:
        console.print(f"[bold red]❌ Error during optimization:[/bold red] {str(e)}")
        return None, None, float('-inf')

def main():
    """
    Main entry point.
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
                        config['backtest_settings'],
                        timeframe
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
                        commission=config['backtest_settings']['commission'],
                        timeframe=timeframe
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