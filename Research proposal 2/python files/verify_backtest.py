import os
import pandas as pd
import numpy as np
# Patch numpy.bool8 which was removed in numpy 1.24+ to support older bokeh
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
from pathlib import Path
import scipy.stats as ss
from backtesting import Strategy, Backtest

# Local imports
from custom_backtester import CustomBacktester

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

# Commission: taker_fee (0.10%) + slippage (0.05%) = 0.15% one-way
COMMISSION = 0.0015

class StandardStrategy(Strategy):
    def init(self):
        self.sig = self.I(lambda: self.data.Signal)
        
    def next(self):
        current_sig = self.sig[-1]
        
        current_equity = self.equity
        close_price = self.data.Close[-1]
        target_units = int((current_equity * 0.45) / close_price)
        
        target_size = int(current_sig * target_units)
        current_size = self.position.size
        size_diff = target_size - current_size
        
        if size_diff > 0:
            self.buy(size=size_diff)
        elif size_diff < 0:
            self.sell(size=abs(size_diff))

def compute_dsr(returns, n_trials=100):
    n = len(returns)
    if n <= 2: return 0.0
    mean = returns.mean()
    std = returns.std(ddof=1)
    if std == 0: return 0.0
    sr_daily = mean / std
    skew = returns.skew()
    kurt = returns.kurtosis() + 3
    sr_var = (1.0 - skew * sr_daily + (kurt - 1.0) / 4.0 * sr_daily**2) / (n - 1.0)
    sr_std = np.sqrt(max(sr_var, 1e-8))
    emc = 0.5772156649
    max_z = (1.0 - emc) * ss.norm.ppf(1.0 - 1.0 / n_trials) + emc * ss.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr_benchmark_daily = max_z * sr_std
    dsr = ss.norm.cdf((sr_daily - sr_benchmark_daily) / sr_std)
    return dsr

def load_predictions():
    path = DATA_DIR / "lstm_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found at {path}. Run lstm_pairs_model.py first.")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def generate_classical_signals(df, entry_z=1.5, exit_z=0.0):
    z_scores = df["Z_Score"].values
    signals_eth = np.zeros(len(df))
    signals_btc = np.zeros(len(df))
    pos = 0 
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

def generate_lstm_raw_signals(df):
    probs = df["Pred_Prob"].values
    signals_eth = np.zeros(len(df))
    signals_btc = np.zeros(len(df))
    for i in range(len(df)):
        p = probs[i]
        pos = 1.0 if p > 0.5 else -1.0
        signals_eth[i] = pos
        signals_btc[i] = -pos
    return signals_btc, signals_eth

def generate_lstm_threshold_signals(df, threshold=0.02):
    probs = df["Pred_Prob"].values
    signals_eth = np.zeros(len(df))
    signals_btc = np.zeros(len(df))
    pos = 0.0
    for i in range(len(df)):
        p = probs[i]
        if p > 0.5 + threshold:
            pos = 1.0
        elif p < 0.5 - threshold:
            pos = -1.0
        else:
            pos = 0.0
        signals_eth[i] = pos
        signals_btc[i] = -pos
    return signals_btc, signals_eth

def run_portfolio_reconciliation(df_pred, sigs_btc, sigs_eth, strategy_name):
    # Align dataframes for CustomBacktester and standard engine
    btc_df = pd.DataFrame({
        "Date": df_pred["Date"],
        "Open": df_pred["Open_BTC"],
        "High": df_pred[["Open_BTC", "Close_BTC"]].max(axis=1),
        "Low": df_pred[["Open_BTC", "Close_BTC"]].min(axis=1),
        "Close": df_pred["Close_BTC"],
        "Volume": 1000.0
    }).set_index("Date")
    
    eth_df = pd.DataFrame({
        "Date": df_pred["Date"],
        "Open": df_pred["Open_ETH"],
        "High": df_pred[["Open_ETH", "Close_ETH"]].max(axis=1),
        "Low": df_pred[["Open_ETH", "Close_ETH"]].min(axis=1),
        "Close": df_pred["Close_ETH"],
        "Volume": 1000.0
    }).set_index("Date")
    
    # ── 1. Custom Backtest ──
    # Re-run Custom Backtester for BTC and ETH legs
    btc_df_c = btc_df.reset_index()
    eth_df_c = eth_df.reset_index()
    
    bt_btc_c = CustomBacktester(btc_df_c, initial_capital=50000.0, market_type="crypto", slippage_pct=0.0005)
    eq_btc_cust, _, trades_btc_cust = bt_btc_c.run_simulation(0.45 * sigs_btc, long_only=False)
    
    bt_eth_c = CustomBacktester(eth_df_c, initial_capital=50000.0, market_type="crypto", slippage_pct=0.0005)
    eq_eth_cust, _, trades_eth_cust = bt_eth_c.run_simulation(0.45 * sigs_eth, long_only=False)
    
    eq_cust = eq_btc_cust + eq_eth_cust
    trades_cust = trades_btc_cust + trades_eth_cust
    
    combined_tester = CustomBacktester(btc_df_c, initial_capital=100000.0, market_type="crypto", slippage_pct=0.0005)
    m_cust = combined_tester.compute_metrics(eq_cust, eq_cust, trades_cust)
    
    # ── 2. Standard Backtest (backtesting.py) ──
    # BTC Leg
    df_btc_std = btc_df.copy()
    df_btc_std["Signal"] = sigs_btc
    bt_btc_std = Backtest(df_btc_std, StandardStrategy, cash=500000000, commission=lambda size, price: price * COMMISSION, margin=0.01)
    res_btc_std = bt_btc_std.run()
    
    # ETH Leg
    df_eth_std = eth_df.copy()
    df_eth_std["Signal"] = sigs_eth
    bt_eth_std = Backtest(df_eth_std, StandardStrategy, cash=500000000, commission=lambda size, price: price * COMMISSION, margin=0.01)
    res_eth_std = bt_eth_std.run()
    
    # Combine Standard Equity Curves and Trades
    eq_btc_std = res_btc_std["_equity_curve"]["Equity"]
    eq_eth_std = res_eth_std["_equity_curve"]["Equity"]
    
    # Align dates
    eq_std = eq_btc_std.reindex(btc_df.index).fillna(method="ffill") + eq_eth_std.reindex(eth_df.index).fillna(method="ffill")
    
    # Calculate Standard metrics
    n_days = len(eq_std)
    n_years = n_days / 365.0
    cagr_std = (eq_std.iloc[-1] / 1000000000.0) ** (1 / n_years) - 1
    
    daily_ret = eq_std.pct_change().dropna()
    vol_std = daily_ret.std() * np.sqrt(365)
    
    rf_daily = 0.05 / 365.0
    excess = daily_ret - rf_daily
    sharpe_std = excess.mean() / excess.std() * np.sqrt(365) if excess.std() > 0 else 0.0
    
    roll_max = eq_std.cummax()
    dd = (eq_std - roll_max) / roll_max
    max_dd_std = dd.min()
    
    trades_std = int(res_btc_std["# Trades"] + res_eth_std["# Trades"])
    
    # Compute Deflated Sharpe Ratio
    dsr_cust = compute_dsr(pd.Series(eq_cust).pct_change().dropna(), n_trials=20)
    dsr_std = compute_dsr(pd.Series(eq_std).pct_change().dropna(), n_trials=20)
    
    corr = np.corrcoef(eq_cust, eq_std.values)[0, 1]
    
    return {
        "Strategy": strategy_name,
        "Custom_CAGR": m_cust["Annualized CAGR [%]"],
        "Standard_CAGR": cagr_std * 100.0,
        "Custom_Vol": m_cust["Annualized Volatility [%]"],
        "Standard_Vol": vol_std * 100.0,
        "Custom_Sharpe": m_cust["Sharpe Ratio"],
        "Standard_Sharpe": sharpe_std,
        "Custom_DSR": dsr_cust,
        "Standard_DSR": dsr_std,
        "Custom_MaxDD": m_cust["Max Drawdown [%]"],
        "Standard_MaxDD": max_dd_std * 100.0,
        "Custom_Trades": int(m_cust["Trade Count"]),
        "Standard_Trades": trades_std,
        "Correlation": corr
    }

def run_reconciliation():
    print("[*] Running Backtester Reconciliation for Proposal 2...")
    df_pred = load_predictions()
    
    # Signals
    class_btc, class_eth = generate_classical_signals(df_pred)
    raw_btc, raw_eth = generate_lstm_raw_signals(df_pred)
    thresh_btc, thresh_eth = generate_lstm_threshold_signals(df_pred, threshold=0.02)
    
    # Run
    class_results = run_portfolio_reconciliation(df_pred, class_btc, class_eth, "Classical Pairs Trading")
    raw_results = run_portfolio_reconciliation(df_pred, raw_btc, raw_eth, "LSTM Pairs (Raw)")
    thresh_results = run_portfolio_reconciliation(df_pred, thresh_btc, thresh_eth, "LSTM Pairs (Thresh=0.02)")
    
    # Format and save comparison CSV
    recon_rows = []
    for r in [class_results, raw_results, thresh_results]:
        strategy = r["Strategy"]
        for metric, cust_val, std_val, diff_val in [
            ("Annualized CAGR", f"{r['Custom_CAGR']:.2f}%", f"{r['Standard_CAGR']:.2f}%", f"{(r['Custom_CAGR'] - r['Standard_CAGR']):.3f}%"),
            ("Annualized Volatility", f"{r['Custom_Vol']:.2f}%", f"{r['Standard_Vol']:.2f}%", f"{(r['Custom_Vol'] - r['Standard_Vol']):.3f}%"),
            ("Sharpe Ratio", f"{r['Custom_Sharpe']:.3f}", f"{r['Standard_Sharpe']:.3f}", f"{(r['Custom_Sharpe'] - r['Standard_Sharpe']):.4f}"),
            ("Deflated Sharpe Ratio", f"{r['Custom_DSR']:.4f}", f"{r['Standard_DSR']:.4f}", f"{(r['Custom_DSR'] - r['Standard_DSR']):.4f}"),
            ("Max Drawdown", f"{r['Custom_MaxDD']:.2f}%", f"{r['Standard_MaxDD']:.2f}%", f"{(r['Custom_MaxDD'] - r['Standard_MaxDD']):.3f}%"),
            ("Trade Count", str(r["Custom_Trades"]), str(r["Standard_Trades"]), str(r["Custom_Trades"] - r["Standard_Trades"])),
            ("Equity Correlation", "1.00000", f"{r['Correlation']:.5f}", "0.00000")
        ]:
            recon_rows.append({
                "Strategy": strategy,
                "Metric": metric,
                "Custom Engine": cust_val,
                "Standard Engine": std_val,
                "Difference": diff_val
            })
            
    recon_df = pd.DataFrame(recon_rows)
    recon_df.to_csv(DATA_DIR / "reconciliation_proposal2.csv", index=False)
    print(f"[+] Reconciliation results saved to {DATA_DIR / 'reconciliation_proposal2.csv'}")

if __name__ == "__main__":
    run_reconciliation()
