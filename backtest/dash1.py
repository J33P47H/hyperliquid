import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================
# Title and Introduction
# =========================
st.title("Ichimokus Backtest Results Dashboard")
st.markdown("""
This dashboard provides an interactive overview of your backtest results for the Ichimokus strategy across multiple coins.
Review key performance metrics, compare runs, and explore how different optimization parameters affect performance.
""")

# =========================
# KPI Dashcards at the Top
# =========================
st.markdown("## Key Performance Indicators")
# (These KPIs are computed on the filtered data later, so for now we use the full data.)
# We will update these once filtering is applied.
kpi_placeholder = st.empty()

# =========================
# Sidebar: Data Upload and Filters
# =========================
st.sidebar.header("Data Input & Filters")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file with backtest results", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# --- Rename Columns for Consistency ---
# Adjust these mappings if your CSV file uses different names.
rename_dict = {
    "Unnamed: 0": "Run Name",
    "Return (Ann.) [%]": "Return (Ann.)",
    "Max. Drawdown [%]": "Max. Drawdown (%)"
}
df.rename(columns=rename_dict, inplace=True)

# -------------------------------
# Sidebar Filters for Top Performers
# -------------------------------
min_trades = st.sidebar.number_input(
    "Minimum number of trades", 
    min_value=1, 
    value=30, 
    step=1, 
    help="Filter out runs with too few trades."
)
min_sharpe = st.sidebar.number_input(
    "Minimum Sharpe Ratio", 
    min_value=0.0, 
    value=1.0, 
    step=0.1, 
    format="%.2f",
    help="Only include runs with a Sharpe Ratio at or above this value."
)
min_return = st.sidebar.number_input(
    "Minimum Annual Return (%)", 
    min_value=-100.0, 
    value=10.0, 
    step=1.0, 
    help="Only include runs with an annual return at or above this percentage."
)

# Apply filters to the DataFrame
filtered_df = df[df["# Trades"] >= min_trades]
filtered_df = filtered_df[filtered_df["Sharpe Ratio"] >= min_sharpe]
filtered_df = filtered_df[filtered_df["Return (Ann.)"] >= min_return]

# Create a new column for color mapping in scatter plot:
# Higher "Drawdown Abs" (i.e. worse drawdown) should be red.
filtered_df["Drawdown Abs"] = -filtered_df["Max. Drawdown (%)"]

# Update the KPI Dashcards based on the filtered data.
if not filtered_df.empty:
    best_return = filtered_df["Return (Ann.)"].max()
    worst_return = filtered_df["Return (Ann.)"].min()
    avg_return = filtered_df["Return (Ann.)"].mean()
    best_sharpe = filtered_df["Sharpe Ratio"].max()
    worst_sharpe = filtered_df["Sharpe Ratio"].min()
    avg_sharpe = filtered_df["Sharpe Ratio"].mean()
    best_drawdown = filtered_df["Max. Drawdown (%)"].min()  # most negative
    avg_drawdown = filtered_df["Max. Drawdown (%)"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Annual Return (%)", f"{best_return:.2f}")
    col2.metric("Worst Annual Return (%)", f"{worst_return:.2f}")
    col3.metric("Average Annual Return (%)", f"{avg_return:.2f}")
    col4.metric("Best Sharpe Ratio", f"{best_sharpe:.2f}")
    
    col5, col6 = st.columns(2)
    col5.metric("Worst Sharpe Ratio", f"{worst_sharpe:.2f}")
    col6.metric("Average Sharpe Ratio", f"{avg_sharpe:.2f}")
    # Drawdown: since values are negative, show the most negative and average.
    col7, col8 = st.columns(2)
    col7.metric("Worst Drawdown (%)", f"{best_drawdown:.2f}")
    col8.metric("Average Drawdown (%)", f"{avg_drawdown:.2f}")
else:
    st.warning("No runs match the current filter criteria.")

# =========================
# Section 0: Return vs Buy & Hold Return Comparison
# =========================
st.header("Return vs Buy & Hold Return Comparison")
# Here we assume the original CSV includes "Return [%]" and "Buy & Hold Return [%]".
if "Return [%]" in filtered_df.columns and "Buy & Hold Return [%]" in filtered_df.columns:
    top10_returns = filtered_df.sort_values("Return [%]", ascending=False).head(10)
    melted = top10_returns.melt(
        id_vars=["Run Name"], 
        value_vars=["Return [%]", "Buy & Hold Return [%]"],
        var_name="Return Type", 
        value_name="Value"
    )
    fig_return = px.bar(
        melted,
        x="Run Name",
        y="Value",
        color="Return Type",
        barmode="group",
        title="Top 10 Runs: Total Return vs Buy & Hold Return",
        labels={"Value": "Return (%)", "Run Name": "Run"}
    )
    st.plotly_chart(fig_return, use_container_width=True)
    st.markdown("""
    **Explanation:** This chart compares the strategy’s total return (*Return [%]*) versus a simple buy‐and‐hold return (*Buy & Hold Return [%]*)
    for the top 10 runs. A larger gap between the two bars for a run indicates that the strategy outperformed a buy‐and‐hold approach.
    """)
else:
    st.info("The uploaded file does not contain columns for 'Return [%]' and 'Buy & Hold Return [%]'.")

# =========================
# Section 1: Raw Data Overview
# =========================
st.header("Raw Data Overview")
st.dataframe(filtered_df.head())
st.markdown("**Explanation:** This table shows the first few rows of the filtered backtest data. Use it to quickly inspect the details of each run.")

# =========================
# Section 2: Summary Table
# =========================
st.subheader("Summary of Key Metrics")
summary_cols = [
    "Run Name", "Return (Ann.)", "Sharpe Ratio", "Sortino Ratio",
    "Calmar Ratio", "Max. Drawdown (%)", "# Trades"
]
st.dataframe(filtered_df[summary_cols].sort_values("Return (Ann.)", ascending=False))
st.markdown("**Explanation:** The summary table lists key performance metrics for each run. It includes annualized return, risk-adjusted ratios, maximum drawdown, and trade count. Sorting by annual return helps highlight top-performing runs.")

# =========================
# Section 3: Scatter Plot: Return (Ann.) vs Sharpe Ratio
# =========================
st.header("Return (Annualized) vs Sharpe Ratio")
# Use the new "Drawdown Abs" column for coloring so that a higher value (worse drawdown) appears red.
fig_scatter = px.scatter(
    filtered_df,
    x="Return (Ann.)",
    y="Sharpe Ratio",
    size="# Trades",
    color="Drawdown Abs",
    hover_data=["Run Name"],
    title="Return vs Sharpe Ratio (Bubble size: # Trades, Color: Absolute Drawdown)",
    color_continuous_scale="RdYlGn_r"  # low values (good) are green; high values (bad) are red.
)
st.plotly_chart(fig_scatter, use_container_width=True)
st.markdown("**Explanation:** In this scatter plot each run is represented as a bubble. The x-axis shows the annualized return and the y-axis the Sharpe ratio. The bubble size reflects the number of trades, and the color (based on the absolute drawdown) indicates risk – runs with higher absolute drawdown (worse) appear red while those with lower drawdown appear green.")

# =========================
# Section 4: Top 10 Runs by Annual Return
# =========================
st.header("Top 10 Runs by Annual Return")
top10 = filtered_df.sort_values("Return (Ann.)", ascending=False).head(10)
fig_bar = px.bar(
    top10,
    x="Run Name",
    y="Return (Ann.)",
    color="Return (Ann.)",
    hover_data=["Sharpe Ratio", "Max. Drawdown (%)", "# Trades"],
    title="Top 10 Runs by Annual Return",
    color_continuous_scale="Viridis",
)
st.plotly_chart(fig_bar, use_container_width=True)
st.markdown("**Explanation:** This bar chart highlights the top 10 runs based on annual return. Hover over the bars to see additional details (such as the Sharpe ratio, maximum drawdown, and trade count) to better assess the performance of these runs.")

# =========================
# Section 5: Box Plots for Performance Metrics
# =========================
st.header("Distribution of Performance Metrics")
metrics = ["Return (Ann.)", "Sharpe Ratio", "Sortino Ratio", "Max. Drawdown (%)", "Profit Factor"]
fig_box = go.Figure()
for metric in metrics:
    fig_box.add_trace(go.Box(y=filtered_df[metric], name=metric))
fig_box.update_layout(title="Distribution of Key Performance Metrics")
st.plotly_chart(fig_box, use_container_width=True)
st.markdown("**Explanation:** The box plots show the distribution of each key performance metric across all runs. They display the median, quartiles, and potential outliers, giving you insight into the variability and overall risk profile of the strategy.")

# =========================
# Section 6: Correlation Heatmap
# =========================
st.header("Correlation Between Key Metrics")
corr_metrics = ["Return (Ann.)", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max. Drawdown (%)", "# Trades", "Profit Factor"]
corr = filtered_df[corr_metrics].corr()
fig_heatmap = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    title="Correlation Heatmap of Performance Metrics"
)
st.plotly_chart(fig_heatmap, use_container_width=True)
st.markdown("**Explanation:** The correlation heatmap illustrates the relationships between different performance metrics. It shows how closely related the metrics are, helping you understand which factors tend to move together and where trade-offs might occur.")

# =========================
# Section 7: Parameter Impact Analysis (Scatter Plot Matrix)
# =========================
st.header("Heatmap of Parameter Combinations")
st.markdown("**Explanation:** Use the dropdowns to select two optimization parameters. The heatmap displays the average annual return for each combination, helping you identify optimal parameter regions.")
param_options = ["param_tenkan_period", "param_kijun_period", "param_senkou_span_b_period", "param_displacement", "param_trailing_stop_pct"]
x_param = st.selectbox("Select X parameter", param_options, index=0)
y_param = st.selectbox("Select Y parameter", param_options, index=1)
heat_data = filtered_df.groupby([x_param, y_param])["Return (Ann.)"].mean().reset_index()
fig_heat_param = px.density_heatmap(
    heat_data,
    x=x_param,
    y=y_param,
    z="Return (Ann.)",
    color_continuous_scale="Viridis",
    title="Heatmap of Average Annual Return by Parameter Combinations"
)
st.plotly_chart(fig_heat_param, use_container_width=True)

# =========================
# Section 8: Scatter Matrix (Pair Plot)
# =========================
st.header("Scatter Matrix (Pair Plot) of Key Metrics")
st.markdown("**Explanation:** The scatter matrix displays pairwise relationships between multiple key performance metrics. Use it to identify potential correlations and patterns among metrics such as annual return, Sharpe ratio, and maximum drawdown.")
pair_cols = ["Return (Ann.)", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Max. Drawdown (%)", "# Trades", "Profit Factor"]
fig_matrix = px.scatter_matrix(
    filtered_df,
    dimensions=pair_cols,
    color="Return (Ann.)",
    title="Scatter Matrix of Key Performance Metrics",
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig_matrix, use_container_width=True)

# =========================
# Section 9: Time Series Plots (Equity Curve)
# =========================
st.header("Equity Curve Time Series")
if "Time" in filtered_df.columns and "Equity" in filtered_df.columns:
    fig_time = px.line(filtered_df, x="Time", y="Equity", title="Equity Curve Over Time")
    st.plotly_chart(fig_time, use_container_width=True)
    st.markdown("**Explanation:** This time series plot shows the evolution of equity over time for the runs (or for a selected run if your CSV is structured that way). It provides insight into the consistency and drawdown behavior over the trading period.")
else:
    st.info("No time series data found. To view equity curves, include 'Time' and 'Equity' columns in your data.")

# =========================
# Conclusion
# =========================
st.markdown("""
### Conclusion
This dashboard provides a comprehensive view of performance and risk metrics, the relationships among key metrics, and the impact of optimization parameters on the strategy's annualized return.
Use the sidebar filters to focus on top-performing runs, and interact with the visualizations to identify robust runs and potential areas for further optimization.
""")
