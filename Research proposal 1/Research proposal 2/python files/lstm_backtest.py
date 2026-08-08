import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from custom_backtester import CustomBacktester

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_predictions():
    path = os.path.join(DATA_DIR, "lstm_predictions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictions not found at {path}. Run lstm_pairs_model.py first.")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df

# Generate Classical Pairs Signals
def generate_classical_signals(df, entry_z=1.5, exit_z=0.0):
    z_scores = df["Z_Score"].values
    signals_eth = np.zeros(len(df))
    signals_btc = np.zeros(len(df))
    
    pos = 0 # 0=cash, 1=long spread (long ETH/short BTC), -1=short spread (short ETH/long BTC)
    
    for i in range(len(df)):
        z = z_scores[i]
        
        if pos == 0:
            if z < -entry_z:
                pos = 1
            elif z > entry_z:
                pos = -1
        elif pos == 1:
            if z >= exit_z:
                pos = 0
        elif pos == -1:
            if z <= exit_z:
                pos = 0
                
        signals_eth[i] = float(pos)
        signals_btc[i] = -float(pos)
        
    return signals_btc, signals_eth

# Generate LSTM Signals (Raw: always in position)
def generate_lstm_raw_signals(df):
    probs = df["Pred_Prob"].values
    signals_eth = np.zeros(len(df))
    signals_btc = np.zeros(len(df))
    
    for i in range(len(df)):
        p = probs[i]
        if p > 0.5:
            pos = 1.0
        else:
            pos = -1.0
        signals_eth[i] = pos
        signals_btc[i] = -pos
        
    return signals_btc, signals_eth

# Generate LSTM Signals (With threshold to filter noise and reduce costs)
def generate_lstm_threshold_signals(df, threshold=0.02):
    probs = df["Pred_Prob"].values
    signals_eth = np.zeros(len(df))
    signals_btc = np.zeros(len(df))
    
    pos = 0.0
    for i in range(len(df)):
        p = probs[i]
        # Entry rules
        if p > 0.5 + threshold:
            pos = 1.0
        elif p < 0.5 - threshold:
            pos = -1.0
        else:
            pos = 0.0 # Go flat in the neutral zone
            
        signals_eth[i] = pos
        signals_btc[i] = -pos
        
    return signals_btc, signals_eth

def run_portfolio_backtest(df_predictions, sigs_btc, sigs_eth, strategy_name):
    # Align dataframes for CustomBacktester
    btc_df = pd.DataFrame({
        "Date": df_predictions["Date"],
        "Open": df_predictions["Open_BTC"],
        "High": df_predictions["Close_BTC"], # Dummy
        "Low": df_predictions["Close_BTC"],  # Dummy
        "Close": df_predictions["Close_BTC"],
        "Volume": 1000.0 # Dummy
    })
    
    eth_df = pd.DataFrame({
        "Date": df_predictions["Date"],
        "Open": df_predictions["Open_ETH"],
        "High": df_predictions["Close_ETH"], # Dummy
        "Low": df_predictions["Close_ETH"],  # Dummy
        "Close": df_predictions["Close_ETH"],
        "Volume": 1000.0 # Dummy
    })
    
    # Backtest individual legs with 50% capital allocation each (Total capital = 100k)
    bt_btc = CustomBacktester(btc_df, initial_capital=50000.0, market_type="crypto", slippage_pct=0.0005)
    eq_btc_net, eq_btc_gross, trades_btc = bt_btc.run_simulation(sigs_btc, long_only=False)
    
    bt_eth = CustomBacktester(eth_df, initial_capital=50000.0, market_type="crypto", slippage_pct=0.0005)
    eq_eth_net, eq_eth_gross, trades_eth = bt_eth.run_simulation(sigs_eth, long_only=False)
    
    # Combine portfolios
    combined_equity = eq_btc_net + eq_eth_net
    combined_equity_gross = eq_btc_gross + eq_eth_gross
    combined_trades = trades_btc + trades_eth
    
    # Compute combined metrics (using BTC timeline/tester with 100k initial capital)
    combined_tester = CustomBacktester(btc_df, initial_capital=100000.0, market_type="crypto", slippage_pct=0.0005)
    metrics = combined_tester.compute_metrics(combined_equity, combined_equity_gross, combined_trades)
    
    return combined_equity, combined_equity_gross, metrics

def plot_proposal2_results(dates, equities, drawdowns, names, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    
    colors = ["#333333", "#1f77b4", "#2ca02c"] # Charcoal, Blue, Green
    
    # Equity curves
    for idx, (equity, name) in enumerate(zip(equities, names)):
        ax1.plot(dates, equity, label=name, color=colors[idx], linewidth=1.5)
    ax1.set_title("Out-of-Sample Performance Comparison (Jan 2023 - May 2026)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    
    # Drawdowns
    for idx, (dd, name) in enumerate(zip(drawdowns, names)):
        ax2.fill_between(dates, dd, 0, color=colors[idx], alpha=0.15)
        ax2.plot(dates, dd, label=name, color=colors[idx], linewidth=1.0)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()

def main():
    print("[*] Loading LSTM out-of-sample predictions...")
    df_pred = load_predictions()
    
    dates = df_pred["Date"].values
    
    # 1. Classical Cointegration Pairs Trading
    print("[*] Simulating Classical Cointegration Pairs Trading baseline...")
    class_sigs_btc, class_sigs_eth = generate_classical_signals(df_pred)
    class_eq, class_eq_gross, class_metrics = run_portfolio_backtest(df_pred, class_sigs_btc, class_sigs_eth, "Classical Cointegration")
    
    # 2. LSTM Pairs Trading (Raw - Always in position)
    print("[*] Simulating LSTM Pairs Trading (Raw, no threshold)...")
    lstm_raw_sigs_btc, lstm_raw_sigs_eth = generate_lstm_raw_signals(df_pred)
    lstm_raw_eq, lstm_raw_eq_gross, lstm_raw_metrics = run_portfolio_backtest(df_pred, lstm_raw_sigs_btc, lstm_raw_sigs_eth, "LSTM Raw")
    
    # 3. LSTM Pairs Trading (Threshold = 0.02)
    print("[*] Simulating LSTM Pairs Trading (Threshold = 0.02)...")
    lstm_thresh_sigs_btc, lstm_thresh_sigs_eth = generate_lstm_threshold_signals(df_pred, threshold=0.02)
    lstm_thresh_eq, lstm_thresh_eq_gross, lstm_thresh_metrics = run_portfolio_backtest(df_pred, lstm_thresh_sigs_btc, lstm_thresh_sigs_eth, "LSTM Threshold (0.02)")
    
    # Compute Drawdown series for plotting
    def compute_dd_series(eq):
        peaks = pd.Series(eq).cummax()
        return (pd.Series(eq) - peaks) / peaks * 100
        
    class_dd = compute_dd_series(class_eq)
    lstm_raw_dd = compute_dd_series(lstm_raw_eq)
    lstm_thresh_dd = compute_dd_series(lstm_thresh_eq)
    
    # Plot results
    print("[*] Generating comparison plots...")
    plot_proposal2_results(
        dates, 
        [class_eq, lstm_raw_eq, lstm_thresh_eq],
        [class_dd, lstm_raw_dd, lstm_thresh_dd],
        ["Classical Pairs Trading", "LSTM Pairs Trading (Raw)", "LSTM Pairs Trading (Threshold=0.02)"],
        "proposal2_performance.png"
    )
    
    # Save metrics csv
    metrics_summary = pd.DataFrame({
        "Metric": list(class_metrics.keys()),
        "Classical Pairs Trading": list(class_metrics.values()),
        "LSTM Pairs (Raw)": list(lstm_raw_metrics.values()),
        "LSTM Pairs (Thresh=0.02)": list(lstm_thresh_metrics.values())
    })
    
    summary_path = os.path.join(DATA_DIR, "proposal2_metrics_summary.csv")
    metrics_summary.to_csv(summary_path, index=False)
    print(f"[+] Metrics summary saved to {summary_path}")
    
    # Print Quantitative Summary Table
    print("\n" + "="*110)
    print(f"{'Strategy':<30} | {'CAGR (%)':<10} | {'Vol (%)':<10} | {'Sharpe':<8} | {'MaxDD (%)':<10} | {'Trades':<8} | {'Hit Rate':<10}")
    print("="*110)
    
    for name, m in [
        ("Classical Pairs Trading", class_metrics),
        ("LSTM Pairs (Raw)", lstm_raw_metrics),
        ("LSTM Pairs (Thresh=0.02)", lstm_thresh_metrics)
    ]:
        print(f"{name:<30} | {m['Annualized CAGR [%]']:<10.2f} | {m['Annualized Volatility [%]']:<10.2f} | {m['Sharpe Ratio']:<8.3f} | {m['Max Drawdown [%]']:<10.2f} | {m['Trade Count']:<8} | {m['Hit Rate [%]']:<10.2f}")
    print("="*110 + "\n")

if __name__ == "__main__":
    main()
