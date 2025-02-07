import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------------
# 1. Load the data from CSV file
# ------------------------------------------------------------------------
# Read the CSV file
df_raw = pd.read_csv('backtest_summary.csv', index_col=0)

# Transpose so each strategy is a row and metrics are columns
df = df_raw.transpose()

# Clean up strategy names by removing timestamp
df['Strategy'] = df.index.map(lambda x: x.split('_')[0])
df.reset_index(drop=True, inplace=True)

# Make numeric columns truly numeric
numeric_columns = [
    'Exposure Time [%]', 'Equity Final [$]', 'Return [%]', 'CAGR [%]',
    'Volatility (Ann.) [%]', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
    'Max. Drawdown [%]', '# Trades', 'Win Rate [%]', 'Profit Factor',
    'Expectancy [%]'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ------------------------------------------------------------------------
# 2. Display the DataFrame
# ------------------------------------------------------------------------
print("DataFrame Head:")
print(df[['Strategy'] + numeric_columns].head(len(df)))

# ------------------------------------------------------------------------
# 3. Visualization Setup
# ------------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Helper function: barplot for a given metric
def plot_bar(metric, title, rotation=45):
    plt.figure()
    order = df.sort_values(by=metric, ascending=False)
    sns.barplot(data=order, x="Strategy", y=metric, palette="viridis")
    plt.title(title)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------
# 4. Key Bar Charts
# ------------------------------------------------------------------------
# A) Final Equity
plot_bar("Equity Final [$]", "Final Equity by Strategy [$]")

# B) Total Return (%)
plot_bar("Return [%]", "Total Return [%] by Strategy")

# C) Sharpe Ratio
plot_bar("Sharpe Ratio", "Sharpe Ratio by Strategy")

# D) Max Drawdown
plot_bar("Max. Drawdown [%]", "Max Drawdown [%] (Higher = Worse)")

# E) Win Rate
plot_bar("Win Rate [%]", "Win Rate [%] by Strategy")

# F) Profit Factor
plot_bar("Profit Factor", "Profit Factor by Strategy")

# ------------------------------------------------------------------------
# 5. Scatter Plot: Return vs. Volatility
# ------------------------------------------------------------------------
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x="Volatility (Ann.) [%]", y="Return [%]", 
                hue="Strategy", palette="tab10", s=100)
plt.title("Return vs. Annual Volatility")
plt.xlabel("Volatility (Ann.) [%]")
plt.ylabel("Return [%]")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------
# 6. Correlation Heatmap among numeric metrics
# ------------------------------------------------------------------------
corr = df[numeric_columns].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap="vlag", fmt=".2f")
plt.title("Correlation Heatmap of Metrics")
plt.tight_layout()
plt.show()
