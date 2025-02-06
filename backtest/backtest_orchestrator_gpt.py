import pandas as pd
from backtesting import Backtest
import json
from datetime import datetime
from pathlib import Path
import importlib
import sys
import itertools
from typing import Dict, List, Any

# Add the strategies directory to Python path
STRATEGIES_DIR = Path("strategies")
DATA_DIR = Path("data/ohlcv")
RESULTS_DIR = Path("data/results")

# Add strategies directory to Python path if it exists
if STRATEGIES_DIR.exists():
    sys.path.append(str(STRATEGIES_DIR.absolute()))

def load_data(csv_path):
    """Loads intraday trading data from CSV with proper timestamp parsing."""
    # Read CSV with timestamp parsing
    df = pd.read_csv(csv_path)
    
    # Convert timestamp column to datetime with a specific format
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    
    # Convert numeric columns to float
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with NaN values
    df = df.dropna()
    
    # Sort explicitly by the 'timestamp' column to ensure chronological order
    df = df.sort_values('timestamp')
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    # Rename columns to match Backtesting.py expected format
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)

    return df

def generate_parameter_combinations(param_config, max_combinations: int = 100) -> List[Dict[str, Any]]:
    """Generate all possible parameter combinations from strategy class configuration."""
    param_ranges = {}
    default_params = {}
    
    # Extract ranges and defaults from param_config
    for param_name, param_info in param_config.items():
        if isinstance(param_info, dict):
            if 'range' in param_info:
                param_ranges[param_name] = param_info['range']
            if 'default' in param_info:
                default_params[param_name] = param_info['default']
    
    if not param_ranges:
        return [default_params] if default_params else []
    
    # Generate all combinations using itertools.product
    keys = list(param_ranges.keys())
    values = [param_ranges[key] for key in keys]
    all_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    # Ensure the default combination is always included
    if default_params not in all_combinations:
        all_combinations.append(default_params)
    
    # If there are more than max_combinations, sample deterministically
    if len(all_combinations) > max_combinations:
        import random
        random.seed(42)  # Fixed seed for reproducibility
        sampled = random.sample(all_combinations, max_combinations - 1)  # Leave room for default params
        if default_params not in sampled:
            sampled.append(default_params)
        all_combinations = sampled
    
    return all_combinations


def run_strategy_backtest(df, strategy_class, strategy_params=None, cash=10_000, commission=0.001):
    """Run backtest for a single strategy."""
    # Use a fresh copy of the DataFrame to avoid any in-place modifications
    df = df.copy()
    # Create Backtest instance
    bt = Backtest(df, strategy_class, cash=cash, commission=commission)
    
    # Run backtest with parameters if provided, otherwise use defaults
    if strategy_params and isinstance(strategy_params, dict):
        stats = bt.run(**strategy_params)
    else:
        stats = bt.run()
        
    return stats, bt

def calculate_buy_and_hold(df):
    """Calculate Buy & Hold return consistently for the entire dataset."""
    first_price = df['Close'].iloc[0]
    last_price = df['Close'].iloc[-1]
    return ((last_price - first_price) / first_price) * 100

def save_results(stats, strategy_name, symbol, timeframe, params=None, output_dir=RESULTS_DIR):
    """Save backtest results to CSV."""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate commissions based on number of trades and final equity
        num_trades = stats['# Trades']
        commission_rate = 0.001  # 0.1%
        avg_trade_size = stats['Equity Final [$]'] / (num_trades if num_trades > 0 else 1)
        total_commissions = num_trades * avg_trade_size * commission_rate if num_trades > 0 else 0
        
        # Create results dictionary with metrics in desired order.
        # Note: The "Buy & Hold Return [%]" here comes from the backtest run.
        # If you need the full-dataset benchmark, consider using calculate_buy_and_hold() separately.
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
        
        # Add parameters if provided
        if params:
            for param_name, param_value in params.items():
                results_dict[f'param_{param_name}'] = param_value
        
        # Convert to DataFrame
        results_df = pd.DataFrame.from_dict(results_dict, orient='index')
        
        # Save to CSV
        filename = output_dir / "backtest_summary.csv"
        print(f"\nSaving results to: {filename}")
        
        # Create unique column name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_column_name = f"{strategy_name}_{symbol}_{timeframe}_{timestamp}"
        results_df.columns = [new_column_name]
        
        if filename.exists():
            try:
                # Load existing results
                existing_df = pd.read_csv(filename, index_col=0)
                
                # Remove any previous runs of the same strategy-symbol combination
                existing_cols = existing_df.columns
                strategy_cols = [col for col in existing_cols if col.startswith(f"{strategy_name}_{symbol}_{timeframe}")]
                if strategy_cols:
                    existing_df = existing_df.drop(columns=strategy_cols)
                
                # Combine existing and new results
                combined_df = pd.concat([existing_df, results_df], axis=1)
                combined_df.to_csv(filename)
                print("Updated existing file")
            except Exception as e:
                print(f"Error appending to file: {str(e)}")
                results_df.to_csv(filename)
                print("Created new file")
        else:
            results_df.to_csv(filename)
            print("Created new file")
        
        return results_df
    
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return None

def load_strategy(strategy_name):
    """Dynamically load strategy class from file."""
    try:
        # Convert strategy name to module name (e.g., 'Ichimoku' -> 'bt_ichimoku_s')
        module_name = f"bt_{strategy_name.lower()}_s"
        module = importlib.import_module(module_name)
        
        # Get the strategy class (assuming it ends with 'Strategy')
        strategy_class = getattr(module, f"{strategy_name}Strategy")
        return strategy_class
    except Exception as e:
        print(f"Error loading strategy {strategy_name}: {str(e)}")
        return None

def load_config(config_file='backtest_config.json'):
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            
            # Update data source paths to use DATA_DIR
            for key, path in config['data_sources'].items():
                config['data_sources'][key] = str(DATA_DIR / Path(path).name)
                
        return config
    except FileNotFoundError:
        print("Config file not found. Using default configuration.")
        return None

def optimize_strategy(df, strategy_name, strategy_class, strategy_config, backtest_settings):
    """Run optimization for a strategy with different parameter combinations."""
    print(f"\nOptimizing {strategy_name} strategy...")
    
    try:
        # Get parameter configuration from strategy class
        param_config = getattr(strategy_class, 'param_config', {})
        if not param_config:
            raise Exception(f"No parameter configuration found for {strategy_name}")

        # Generate parameter combinations using the strategy's param_config
        param_combinations = generate_parameter_combinations(
            param_config,
            backtest_settings.get('max_combinations', 100)
        )
        
        best_stats = None
        best_params = None
        best_metric_value = float('-inf')
        optimization_metric = backtest_settings.get('optimization_metric', 'Sharpe Ratio')
        
        total_combinations = len(param_combinations)
        print(f"Testing {total_combinations} parameter combinations...")
        
        import math  # for isnan check
        
        for i, params in enumerate(param_combinations, 1):
            try:
                # Run backtest with the current parameters.
                stats, _ = run_strategy_backtest(
                    df,
                    strategy_class,
                    params,
                    cash=backtest_settings['initial_capital'],
                    commission=backtest_settings['commission']
                )
                
                # Skip this combination if no trades were made.
                if stats['# Trades'] == 0:
                    continue
                
                current_metric = stats[optimization_metric]
                # Skip if the metric is None or NaN.
                if current_metric is None or (isinstance(current_metric, float) and math.isnan(current_metric)):
                    continue
                
                if current_metric > best_metric_value:
                    best_metric_value = current_metric
                    best_stats = stats
                    best_params = params
                    print(f"\nNew best {optimization_metric}: {best_metric_value}")
                    print(f"Parameters: {best_params}")
                
                # Print progress every 10%
                if i % max(1, total_combinations // 10) == 0:
                    print(f"Progress: {i}/{total_combinations} combinations tested ({i/total_combinations*100:.1f}%)")
                    
            except Exception as e:
                print(f"Warning: Failed to test parameter combination {params}: {str(e)}")
                continue
        
        if best_stats is None:
            raise Exception("No valid parameter combinations found")
            
        print(f"\nOptimization completed. Best {optimization_metric}: {best_metric_value}")
        print(f"Best parameters: {best_params}")
        return best_stats, best_params, best_metric_value
        
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        return None, None, float('-inf')


def main():
    # Create necessary directories
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load configuration
        config = load_config()
        if not config:
            return
        
        # Get enabled strategies
        enabled_strategies = {
            name: info for name, info in config['strategies'].items()
            if info.get('enabled', True)
        }
        
        print(f"\nEnabled strategies: {list(enabled_strategies.keys())}")
        
        # Run backtests for each data source
        for symbol_tf, csv_path in config['data_sources'].items():
            symbol, timeframe = symbol_tf.split('_')
            print(f"\nTesting on {symbol} ({timeframe})")
            
            try:
                # Load data
                df = load_data(csv_path)
                print(f"Loaded data shape: {df.shape}")
                
                # Run each enabled strategy
                for strategy_name, strategy_info in enabled_strategies.items():
                    print(f"\nRunning {strategy_name} strategy...")
                    
                    try:
                        # Load strategy class
                        strategy_class = load_strategy(strategy_name)
                        if not strategy_class:
                            continue
                        
                        if strategy_info.get('optimize', False):
                            # Run optimization
                            best_stats, best_params, best_metric = optimize_strategy(
                                df,
                                strategy_name,
                                strategy_class,
                                strategy_info,
                                config['backtest_settings']
                            )
                            
                            if best_metric != float('-inf') and best_params:
                                # Save results with best parameters
                                saved_df = save_results(
                                    best_stats,
                                    strategy_name,
                                    symbol,
                                    timeframe,
                                    params=best_params
                                )
                                
                                if saved_df is not None:
                                    print(f"{strategy_name} optimization results saved successfully")
                            else:
                                print(f"Warning: Optimization failed for {strategy_name}")
                        else:
                            # Get default parameters from strategy class
                            param_config = getattr(strategy_class, 'param_config', {})
                            default_params = {
                                name: config['default']
                                for name, config in param_config.items()
                                if isinstance(config, dict) and 'default' in config
                            }
                            
                            # Run single backtest with default parameters
                            stats, _ = run_strategy_backtest(
                                df,
                                strategy_class,
                                default_params,
                                cash=config['backtest_settings']['initial_capital'],
                                commission=config['backtest_settings']['commission']
                            )
                            
                            # Save results
                            saved_df = save_results(
                                stats, 
                                strategy_name, 
                                symbol, 
                                timeframe, 
                                params=default_params
                            )
                            if saved_df is not None:
                                print(f"{strategy_name} results saved successfully")
                        
                        print(f"{strategy_name} completed successfully")
                        
                    except Exception as e:
                        print(f"Error running {strategy_name}: {str(e)}")
                        continue
                    
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return

if __name__ == "__main__":
    main()
