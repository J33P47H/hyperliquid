import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(file_path):
    """
    Load the results CSV (with metrics as rows and run names as columns)
    and transpose it so that each run becomes a row.
    """
    df = pd.read_csv(file_path, index_col=0)
    df = df.transpose()
    return df


def parse_run_name(run_name):
    """
    Parse a run name (e.g., "RSIDivergence_SOL_1h_20250205_160139") into its components.
    Returns: strategy, symbol, timeframe, and a timestamp (as datetime).
    Assumes the run name is composed of 5 parts separated by underscores.
    """
    parts = run_name.split('_')
    if len(parts) >= 5:
        strategy = parts[0]
        symbol = parts[1]
        timeframe = parts[2]
        # Combine the date and time parts
        date_str = parts[3]
        time_str = parts[4]
        try:
            ts = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        except Exception:
            ts = None
    else:
        strategy, symbol, timeframe, ts = run_name, "", "", None
    return strategy, symbol, timeframe, ts


def add_run_metadata(df):
    """
    Parse the index (run names) to add columns for Strategy, Symbol, Timeframe, and Timestamp.
    """
    strategies, symbols, timeframes, timestamps = [], [], [], []
    for run in df.index:
        strat, sym, tf, ts = parse_run_name(run)
        strategies.append(strat)
        symbols.append(sym)
        timeframes.append(tf)
        timestamps.append(ts)
    df['Strategy'] = strategies
    df['Symbol'] = symbols
    df['Timeframe'] = timeframes
    df['Timestamp'] = timestamps
    return df


def convert_metrics(df):
    """
    Convert a selected list of performance metric columns to numeric,
    so that plotting and calculations work correctly.
    """
    metrics = [
        'Equity Final [$]', 'Return [%]', 'Return (Ann.) [%]', 'Sharpe Ratio',
        'Sortino Ratio', 'Calmar Ratio', '# Trades', 'Win Rate [%]',
        'Profit Factor', 'Expectancy [%]', 'SQN', 'Kelly Criterion'
    ]
    for metric in metrics:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
    return df


def best_performers_table(df, metric='Return [%]', top_n=5):
    """
    Print a table of the top performers (runs) sorted by the given metric.
    """
    best = df.sort_values(metric, ascending=False).head(top_n)
    print(f"\nTop {top_n} performers by {metric}:")
    print(best[['Strategy', 'Symbol', 'Timeframe', metric]])


def plot_return_bar(df):
    """
    Create a horizontal bar chart of Return [%] for each run,
    colored by strategy.
    """
    # Sort by return
    df_sorted = df.sort_values('Return [%]', ascending=True)
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(
        x='Return [%]', y=df_sorted.index, data=df_sorted,
        hue='Strategy', dodge=False, palette="viridis"
    )
    plt.title('Return [%] by Run')
    plt.xlabel('Return [%]')
    plt.ylabel('Run (ID)')
    plt.tight_layout()
    plt.show()


def plot_sharpe_vs_return(df):
    """
    Create a scatter plot of Sharpe Ratio vs Return [%],
    using shape or color to denote the strategy.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x='Sharpe Ratio', y='Return [%]', hue='Strategy',
        style='Timeframe', s=100, palette='deep'
    )
    plt.title('Sharpe Ratio vs Return [%]')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Return [%]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_boxplot_by_strategy(df):
    """
    Create a boxplot showing the distribution of returns grouped by strategy.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Strategy', y='Return [%]', data=df, palette='Set3')
    plt.title('Distribution of Returns by Strategy')
    plt.tight_layout()
    plt.show()


def correlation_heatmap(df):
    """
    Create a heatmap of correlations among key performance metrics.
    """
    # Select a subset of performance metrics for the heatmap
    metrics = [
        'Return [%]', 'Return (Ann.) [%]', 'Sharpe Ratio',
        'Sortino Ratio', 'Calmar Ratio', 'Win Rate [%]',
        'Profit Factor', 'Expectancy [%]', 'SQN', 'Kelly Criterion'
    ]
    data = df[metrics].dropna()
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Performance Metrics')
    plt.tight_layout()
    plt.show()


def main():
    # Set the path to your results CSV file (adjust if needed)
    file_path = os.path.join('data', 'results', 'backtest_summary.csv')
    if not os.path.exists(file_path):
        print(f"Results file not found at {file_path}")
        return

    # Load and process the results
    df = load_results(file_path)
    df = add_run_metadata(df)
    df = convert_metrics(df)

    # Show a quick summary table (first few rows)
    print("Summary (first 5 runs):")
    print(df.head())

    # Print best performers by Return and Sharpe Ratio
    best_performers_table(df, metric='Return [%]', top_n=5)
    best_performers_table(df, metric='Sharpe Ratio', top_n=5)

    # Create visualizations
    plot_return_bar(df)
    plot_sharpe_vs_return(df)
    plot_boxplot_by_strategy(df)
    correlation_heatmap(df)


if __name__ == '__main__':
    main()
