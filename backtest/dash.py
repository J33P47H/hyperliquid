import os
from datetime import datetime
# AAAAA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Use st.cache_data for data caching (st.cache is deprecated)
@st.cache_data
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
    Parse a run name (e.g., "RSIDivergence_SOL_1h_20250205_160139")
    into its components: strategy, symbol, timeframe, and a timestamp.
    Assumes the name contains at least 5 underscore-separated parts.
    """
    parts = run_name.split('_')
    if len(parts) >= 5:
        strategy = parts[0]
        symbol = parts[1]
        timeframe = parts[2]
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
    Add metadata columns (Strategy, Symbol, Timeframe, Timestamp) to the dataframe.
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
    Convert selected performance metric columns to numeric.
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


# ------------------- Load and Process Data -------------------
# Adjust the path to your CSV file if needed.
results_file = os.path.join('data', 'results', 'backtest_summary.csv')
if not os.path.exists(results_file):
    st.error(f"Results file not found at {results_file}")
    st.stop()

df = load_results(results_file)
df = add_run_metadata(df)
df = convert_metrics(df)

# ------------------- Streamlit Dashboard Layout -------------------
st.title("Backtest Results Dashboard")
st.markdown(
    """
    This dashboard shows performance results for your backtests.
    Use the sidebar to filter by strategy, symbol, and timeframe.
    **Note:** To view this app properly, run it with:
    
        streamlit run dash.py
    """
)

# Sidebar filters
st.sidebar.header("Filters")
strategy_options = df['Strategy'].dropna().unique().tolist()
symbol_options = df['Symbol'].dropna().unique().tolist()
timeframe_options = df['Timeframe'].dropna().unique().tolist()

selected_strategies = st.sidebar.multiselect("Select Strategy", options=strategy_options, default=strategy_options)
selected_symbols = st.sidebar.multiselect("Select Symbol", options=symbol_options, default=symbol_options)
selected_timeframes = st.sidebar.multiselect("Select Timeframe", options=timeframe_options, default=timeframe_options)

# Apply filters
filtered_df = df[
    (df['Strategy'].isin(selected_strategies)) &
    (df['Symbol'].isin(selected_symbols)) &
    (df['Timeframe'].isin(selected_timeframes))
]

st.write("### Filtered Results")
st.dataframe(filtered_df.reset_index(drop=True))

# ------------------- Top Performers -------------------
st.sidebar.header("Top Performers")
ranking_metric = st.sidebar.selectbox("Select Metric for Ranking", options=['Return [%]', 'Sharpe Ratio', 'Sortino Ratio'])
top_n = st.sidebar.slider("Top N Runs", 1, 20, 5)

top_performers = filtered_df.sort_values(ranking_metric, ascending=False).head(top_n)
st.write(f"### Top {top_n} Performers by {ranking_metric}")
st.table(top_performers[['Strategy', 'Symbol', 'Timeframe', ranking_metric]])

# ------------------- Visualizations -------------------
st.header("Visualizations")

# Plot 1: Horizontal Bar Chart for Return [%]
st.subheader("Return [%] by Run")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sorted_df = filtered_df.sort_values("Return [%]", ascending=True)
sns.barplot(x="Return [%]", y=sorted_df.index, data=sorted_df, hue="Strategy", dodge=False, palette="viridis", ax=ax1)
ax1.set_xlabel("Return [%]")
ax1.set_ylabel("Run ID")
ax1.set_title("Return [%] for Each Run")
plt.tight_layout()
st.pyplot(fig1)

# Plot 2: Scatter Plot of Sharpe Ratio vs Return [%]
st.subheader("Sharpe Ratio vs Return [%]")
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=filtered_df, x="Sharpe Ratio", y="Return [%]", hue="Strategy", style="Timeframe", s=100, palette="deep", ax=ax2)
ax2.set_title("Sharpe Ratio vs Return [%]")
ax2.set_xlabel("Sharpe Ratio")
ax2.set_ylabel("Return [%]")
ax2.grid(True)
plt.tight_layout()
st.pyplot(fig2)

# Plot 3: Boxplot of Return [%] by Strategy
st.subheader("Distribution of Return [%] by Strategy")
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.boxplot(x="Strategy", y="Return [%]", data=filtered_df, hue="Strategy", palette="Set3", ax=ax3)
# Remove legend if it exists
lgd = ax3.get_legend()
if lgd is not None:
    lgd.remove()
ax3.set_title("Return [%] Distribution by Strategy")
plt.tight_layout()
st.pyplot(fig3)

# Plot 4: Correlation Heatmap of Key Metrics
st.subheader("Correlation Heatmap of Performance Metrics")
heat_metrics = [
    'Return [%]', 'Return (Ann.) [%]', 'Sharpe Ratio', 'Sortino Ratio',
    'Calmar Ratio', 'Win Rate [%]', 'Profit Factor', 'Expectancy [%]', 'SQN', 'Kelly Criterion'
]
heat_data = filtered_df[heat_metrics].dropna()
fig4, ax4 = plt.subplots(figsize=(10, 8))
sns.heatmap(heat_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
ax4.set_title("Correlation Heatmap")
plt.tight_layout()
st.pyplot(fig4)

st.markdown(f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
