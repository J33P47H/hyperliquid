import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from pathlib import Path

def load_results(file_path='data/results/backtest_summary.csv'):
    """Load and preprocess the backtest results."""
    # Read the CSV file with the first column as index
    df = pd.read_csv(file_path, index_col=0)
    
    # Clean up strategy names in columns
    df.columns = [col.split('_')[0] for col in df.columns]
    
    return df

def create_performance_comparison(df):
    """Create a performance comparison plot."""
    metrics = ['Return [%]', 'Sharpe Ratio', 'Max. Drawdown [%]', 'Win Rate [%]']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metrics,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Add traces for each metric
    for i, metric in enumerate(metrics, 1):
        row = (i-1) // 2 + 1
        col = (i-1) % 2 + 1
        
        # Get values for each strategy
        values = df.loc[metric]
        strategies = values.index
        
        # Convert to numeric, replacing any errors with 0
        values = pd.to_numeric(values, errors='coerce').fillna(0)
        
        fig.add_trace(
            go.Bar(
                x=list(values.index),
                y=values.values,
                name=metric,
                text=values.round(2),
                textposition='auto',
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=800,
        title_text="Strategy Performance Comparison",
        showlegend=False,
        template='plotly_white'
    )
    
    # Update axes labels
    for i in range(len(metrics)):
        row = i // 2 + 1
        col = i % 2 + 1
        fig.update_xaxes(tickangle=45, row=row, col=col)
    
    return fig

def create_risk_reward_scatter(df):
    """Create a risk-reward scatter plot."""
    # Convert values to numeric, replacing any errors with 0
    returns = pd.to_numeric(df.loc['Return [%]'], errors='coerce').fillna(0)
    drawdowns = pd.to_numeric(df.loc['Max. Drawdown [%]'], errors='coerce').fillna(0)
    sharpe = pd.to_numeric(df.loc['Sharpe Ratio'], errors='coerce').fillna(0)
    trades = pd.to_numeric(df.loc['# Trades'], errors='coerce').fillna(0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdowns,
        y=returns,
        mode='markers+text',
        text=returns.index,
        textposition="top center",
        marker=dict(
            size=(sharpe + 1) * 5,  # Size based on Sharpe ratio (add 1 to handle negative values)
            color=trades,     # Color based on number of trades
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Number of Trades")
        ),
        hovertemplate="<b>%{text}</b><br>" +
                     "Return: %{y:.2f}%<br>" +
                     "Max Drawdown: %{x:.2f}%<br>" +
                     "Sharpe Ratio: %{marker.size:.2f}<br>" +
                     "Trades: %{marker.color}<br>" +
                     "<extra></extra>"
    ))
    
    fig.update_layout(
        title="Risk-Reward Analysis",
        xaxis_title="Maximum Drawdown (%)",
        yaxis_title="Return (%)",
        template='plotly_white',
        height=600
    )
    
    return fig

def create_trade_analysis(df):
    """Create trade analysis plots."""
    metrics = ['# Trades', 'Win Rate [%]', 'Avg. Trade [%]', 'Profit Factor']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=metrics,
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    for i, metric in enumerate(metrics, 1):
        row = (i-1) // 2 + 1
        col = (i-1) % 2 + 1
        
        # Convert values to numeric, replacing any errors with 0
        values = pd.to_numeric(df.loc[metric], errors='coerce').fillna(0)
        
        fig.add_trace(
            go.Bar(
                x=list(values.index),
                y=values.values,
                name=metric,
                text=values.round(2),
                textposition='auto',
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=800,
        title_text="Trade Analysis",
        showlegend=False,
        template='plotly_white'
    )
    
    # Update axes labels
    for i in range(len(metrics)):
        row = i // 2 + 1
        col = i % 2 + 1
        fig.update_xaxes(tickangle=45, row=row, col=col)
    
    return fig

def create_parameter_heatmap(df):
    """Create a heatmap of optimized parameters."""
    # Get all parameter columns
    param_cols = [col for col in df.index if col.startswith('param_')]
    if not param_cols:
        return None
        
    # Create a parameter matrix
    param_matrix = df.loc[param_cols]
    
    # Convert to numeric, replacing any errors with 0
    param_matrix = param_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=param_matrix.values,
        x=param_matrix.columns,
        y=param_matrix.index,
        colorscale='RdBu',
        text=param_matrix.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Parameter Optimization Results",
        xaxis_title="Strategy",
        yaxis_title="Parameter",
        height=max(400, len(param_cols) * 30),
        template='plotly_white'
    )
    
    return fig

def main():
    # Create results directory if it doesn't exist
    results_dir = Path('data/results/visualizations')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the results
    df = load_results()
    
    # Create visualizations
    figs = {
        'performance': create_performance_comparison(df),
        'risk_reward': create_risk_reward_scatter(df),
        'trade_analysis': create_trade_analysis(df),
        'parameters': create_parameter_heatmap(df)
    }
    
    # Save all figures as HTML files
    for name, fig in figs.items():
        if fig is not None:
            output_file = results_dir / f"{name}.html"
            fig.write_html(str(output_file))
            print(f"Saved {name} visualization to {output_file}")

if __name__ == "__main__":
    main() 