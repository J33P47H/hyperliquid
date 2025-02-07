import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Title and Description ---
st.title("Ichimokus Backtest Results Dashboard")
st.markdown("""
This dashboard provides an interactive overview of your backtest results for the Ichimokus strategy across multiple coins.
You can review key performance metrics, compare runs, and explore how different optimization parameters affect performance.
""")

# --- Sidebar: Data Upload and Filters ---
st.sidebar.header("Data Input & Filters")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file with backtest results", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# --- Rename Columns for Consistency ---
# These mappings convert some of your CSV's column names to standardized names used in this dashboard.
rename_dict = {
    "Unnamed: 0": "Run Name",
    "Return (Ann.) [%]": "Return (Ann.)",
    "Max. Drawdown [%]": "Max. Drawdown (%)"
}
df.rename(columns=rename_dict, inplace=True)

# --- Filter Data by Minimum Number of Trades ---
min_trades = st.sidebar.number_input(
    "Minimum number of trades", 
    min_value=1, 
    value=30, 
    step=1, 
    help="Set a minimum number of trades to filter out runs with too few samples."
)
filtered_df = df[df["# Trades"] >= min_trades]

# --- New Section: Compare Return vs Buy & Hold Return ---
st.header("Return vs Buy & Hold Return Comparison")
# We'll select the top 10 runs based on the overall Return [%] (column 8 in your CSV)
# Note: This uses the columns as they appear in the CSV: "Return [%]" and "Buy & Hold Return [%]".
top10_returns = filtered_df.sort_values("Return [%]", ascending=False).head(10)
# Reshape the DataFrame so we can plot grouped bars for each run.
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
**Explanation:** This chart compares the total return achieved by the strategy (*Return [%]*) versus a simple buy‐and‐hold return (*Buy & Hold Return [%]*)
for the top 10 runs. A larger gap between the two bars for a given run indicates that the strategy outperformed a buy‐and‐hold approach.
""")

# --- Section 1: Raw Data Overview ---
st.header("Raw Data Overview")
st.dataframe(filtered_df.head())
st.markdown("**Explanation:** This table shows the first few rows of the filtered backtest data. Use it to quickly inspect the details of each run.")

# --- Section 2: Summary Table ---
st.subheader("Summary of Key Metrics")
summary_cols = [
    "Run Name", "Return (Ann.)", "Sharpe Ratio", "Sortino Ratio",
    "Calmar Ratio", "Max. Drawdown (%)", "# Trades"
]
st.dataframe(filtered_df[summary_cols].sort_values("Return (Ann.)", ascending=False))
st.markdown("**Explanation:** The summary table lists key performance metrics for each run. It includes annualized return, risk-adjusted ratios, maximum drawdown, and trade count. Sorting by annual return helps highlight top-performing runs.")

# --- Section 3: Scatter Plot: Return vs Sharpe Ratio ---
st.header("Return (Annualized) vs Sharpe Ratio")
fig_scatter = px.scatter(
    filtered_df,
    x="Return (Ann.)",
    y="Sharpe Ratio",
    size="# Trades",
    color="Max. Drawdown (%)",
    hover_data=["Run Name"],
    title="Return vs Sharpe Ratio (Bubble size: # Trades, Color: Max Drawdown %)",
    color_continuous_scale="RdYlGn_r",
)
st.plotly_chart(fig_scatter, use_container_width=True)
st.markdown("**Explanation:** This scatter plot displays each run as a bubble. The x-axis shows the annualized return, the y-axis shows the Sharpe ratio, the bubble size reflects the number of trades, and the bubble color indicates the maximum drawdown. This visualization helps you identify runs that combine high returns with good risk-adjusted performance.")

# --- Section 4: Top 10 Runs by Annual Return ---
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

# --- Section 5: Box Plots for Performance Metrics ---
st.header("Distribution of Performance Metrics")
metrics = ["Return (Ann.)", "Sharpe Ratio", "Sortino Ratio", "Max. Drawdown (%)", "Profit Factor"]
fig_box = go.Figure()
for metric in metrics:
    fig_box.add_trace(go.Box(y=filtered_df[metric], name=metric))
fig_box.update_layout(title="Distribution of Key Performance Metrics")
st.plotly_chart(fig_box, use_container_width=True)
st.markdown("**Explanation:** The box plots show the distribution of each key performance metric across all runs. They display the median, quartiles, and potential outliers, giving you insight into the variability and overall risk profile of the strategy.")

# --- Section 6: Correlation Heatmap ---
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

# --- Section 7: Parameter Impact Analysis ---
st.header("Impact of Optimization Parameters on Performance")
st.markdown("Below, explore how different optimization parameters affect the annual return.")
param_cols = [
    "param_tenkan_period", 
    "param_kijun_period", 
    "param_senkou_span_b_period", 
    "param_displacement", 
    "param_trailing_stop_pct"
]
for param in param_cols:
    fig_param = px.scatter(
        filtered_df,
        x=param,
        y="Return (Ann.)",
        hover_data=["Run Name", "Sharpe Ratio"],
        title=f"Return (Ann.) vs {param}",
    )
    st.plotly_chart(fig_param, use_container_width=True)
    st.markdown(f"**Explanation:** This chart shows the relationship between the parameter **{param}** and the annual return. It helps you evaluate how varying this parameter can impact the strategy's performance.")

# --- Conclusion ---
st.markdown("""
### Conclusion
This dashboard provides a comprehensive view of performance and risk metrics, the relationships among key metrics, and the impact of optimization parameters on the strategy's annualized return.
Use the sidebar to filter the results and interact with the visualizations to identify robust runs and potential areas for further optimization.
""")
