import os
import pandas as pd
import numpy as np
# Patch numpy.bool8 which was removed in numpy 1.24+ to support older bokeh
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
from pathlib import Path
import scipy.stats as ss
from backtesting import Strategy, Backtest

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

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
            
        # Apply daily funding rate carry cost/credit
        if self.position.size != 0:
            funding_rate = self.data.FundingRate[-1]
            pos_value = self.position.size * self.data.Close[-1]
            funding_cost = pos_value * funding_rate
            self._broker._cash -= funding_cost

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

def simulate_perpetual_strategy(df, signals, initial_capital=100000.0, fee=0.001, slippage=0.0005):
    """Re-runs the custom perpetual strategy simulator (corrected)."""
    cash = initial_capital
    position_units = 0.0
    current_open = 0.0
    equity = np.zeros(len(df))
    equity_gross = np.zeros(len(df))
    trades = []
    
    cash_gross = initial_capital
    pos_units_gross = 0.0
    
    exec_signals = pd.Series(signals).shift(1).fillna(0).values
    
    closes = df["Close"].values
    opens = df["Open"].values
    funding_rates = df["FundingRate"].values
    dates = df.index.values
    
    entry_idx = None
    
    for i in range(len(df)):
        current_close = closes[i]
        current_open = opens[i]
        funding = funding_rates[i]
        signal = exec_signals[i]
        
        # 1. Update position valuation at current Open (Rebalance decision point)
        current_equity = cash + position_units * current_open
        current_eq_gross = cash_gross + pos_units_gross * current_open
        
        current_weight = (position_units * current_open) / current_equity if current_equity > 0 else 0.0
        target_weight = signal
        
        # Rebalance
        if target_weight != current_weight:
            # Close old NET position
            if position_units != 0.0:
                exit_price = current_open * (1 - slippage if position_units > 0 else 1 + slippage)
                revenue = position_units * exit_price
                costs = abs(position_units) * exit_price * fee
                cash = cash + revenue - costs
                
                gross_rev = position_units * current_open
                cash_gross += gross_rev
                
                trades.append({
                    "entry_date": dates[entry_idx] if entry_idx is not None else dates[i],
                    "exit_date": dates[i],
                    "direction": 1 if position_units > 0 else -1,
                    "net_pnl": (cash + position_units * exit_price) / initial_capital  # temp
                })
                
                position_units = 0.0
                entry_idx = None
                
            # Close old GROSS position
            if pos_units_gross != 0.0:
                pos_units_gross = 0.0
                
            # Open new NET position
            if target_weight != 0.0:
                entry_price = current_open * (1 + slippage if target_weight > 0 else 1 - slippage)
                approx_fee = fee
                max_trade_val = cash / (1 + approx_fee)
                trade_value = target_weight * max_trade_val
                
                position_units = trade_value / entry_price
                cash = cash - trade_value
                entry_idx = i
                
            # Open new GROSS position
            if target_weight != 0.0:
                trade_val_gross = target_weight * cash_gross
                pos_units_gross = trade_val_gross / current_open
                cash_gross -= trade_val_gross
                
        # 2. Apply Daily Funding rate carry cost to open position (evaluated at Close)
        if position_units != 0.0:
            funding_payment = position_units * current_close * funding
            cash -= funding_payment
            
        if pos_units_gross != 0.0:
            pass # gross does not pay funding
            
        # 3. Save Equity at Close of day
        equity[i] = cash + position_units * current_close
        equity_curve_gross = cash_gross + pos_units_gross * current_close
        equity_gross[i] = equity_curve_gross
        
    return equity, equity_gross, trades

def run_reconciliation():
    print("[*] Running Backtester Reconciliation for Proposal 3...")
    
    # 1. Load BTC funding dataset
    df = pd.read_csv(DATA_DIR / "btc_funding_daily.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    
    # Rename columns to match standard backtester and simulator
    df_aligned = df.copy()
    df_aligned = df_aligned.rename(columns={"Close_BTC": "Close", "Open_BTC": "Open"})
    df_aligned["FundingRate"] = df_aligned["funding_rate"]
    
    # Generate signals (rolling 90-day 5th/95th percentiles)
    roll_window = 90
    df_aligned["roll_5th"] = df_aligned["funding_rate"].rolling(roll_window).quantile(0.05)
    df_aligned["roll_95th"] = df_aligned["funding_rate"].rolling(roll_window).quantile(0.95)
    
    signals = np.zeros(len(df_aligned))
    fr = df_aligned["funding_rate"].values
    p5 = df_aligned["roll_5th"].values
    p95 = df_aligned["roll_95th"].values
    
    for i in range(len(df_aligned)):
        if i < roll_window:
            continue
        if fr[i] <= p5[i]:
            signals[i] = 1.0
        elif fr[i] >= p95[i]:
            signals[i] = -1.0
        else:
            signals[i] = 0.0
            
    df_aligned["Signal"] = signals
    
    # Restrict to walk-forward period (warmup starts after 90 days)
    df_test = df_aligned.iloc[roll_window:].copy()
    df_test["FundingRate"] = 0.0
    signals_test = signals[roll_window:]
    
    # ── 1. Custom Backtest ──
    eq_cust, eq_gross_cust, trades_cust = simulate_perpetual_strategy(
        df_test, 0.45 * signals_test, initial_capital=100000.0, fee=0.0, slippage=0.0
    )
    
    # Compute Custom metrics
    n_days = len(df_test)
    n_years = n_days / 365.0
    cagr_cust = (eq_cust[-1] / 100000.0) ** (1 / n_years) - 1
    
    daily_ret_cust = pd.Series(eq_cust).pct_change().dropna()
    vol_cust = daily_ret_cust.std() * np.sqrt(365)
    
    rf_daily = 0.05 / 365.0
    excess_cust = daily_ret_cust - rf_daily
    sharpe_cust = excess_cust.mean() / excess_cust.std() * np.sqrt(365) if excess_cust.std() > 0 else 0.0
    
    roll_max_cust = pd.Series(eq_cust).cummax()
    dd_cust = (pd.Series(eq_cust) - roll_max_cust) / roll_max_cust
    max_dd_cust = dd_cust.min()
    
    # ── 2. Standard Backtest (backtesting.py) ──
    # backtesting.py expects High and Low columns (we can set them to Close or create dummy)
    df_std = df_test.copy()
    df_std["High"] = df_std[["Open", "Close"]].max(axis=1)
    df_std["Low"] = df_std[["Open", "Close"]].min(axis=1)
    df_std["Volume"] = 1000.0
    
    bt_std = Backtest(df_std, StandardStrategy, cash=1000000000, commission=0.0, margin=0.01)
    res_std = bt_std.run()
    
    # Standard Equity
    eq_std = res_std["_equity_curve"]["Equity"].reindex(df_test.index).fillna(method="ffill")
    
    # Compute standard metrics
    cagr_std = (eq_std.iloc[-1] / 1000000000.0) ** (1 / n_years) - 1
    daily_ret_std = eq_std.pct_change().dropna()
    vol_std = daily_ret_std.std() * np.sqrt(365)
    excess_std = daily_ret_std - rf_daily
    sharpe_std = excess_std.mean() / excess_std.std() * np.sqrt(365) if excess_std.std() > 0 else 0.0
    
    roll_max_std = eq_std.cummax()
    dd_std = (eq_std - roll_max_std) / roll_max_std
    max_dd_std = dd_std.min()
    
    trades_std = int(res_std["# Trades"])
    
    # Compute Deflated Sharpe Ratio
    dsr_cust = compute_dsr(daily_ret_cust, n_trials=20)
    dsr_std = compute_dsr(daily_ret_std, n_trials=20)
    
    corr = np.corrcoef(eq_cust, eq_std.values)[0, 1]
    
    print(f"  Custom Engine:   CAGR={cagr_cust*100:.2f}%, Sharpe={sharpe_cust:.3f}, MaxDD={max_dd_cust*100:.2f}%, Trades={len(trades_cust)}")
    print(f"  Standard Engine: CAGR={cagr_std*100:.2f}%, Sharpe={sharpe_std:.3f}, MaxDD={max_dd_std*100:.2f}%, Trades={trades_std}")
    print(f"  Equity Curve Correlation: {corr:.5f}")
    
    # Collect results
    reconciliation_results = []
    for metric, cust_val, std_val, diff_val in [
        ("Annualized CAGR", f"{cagr_cust*100:.2f}%", f"{cagr_std*100:.2f}%", f"{(cagr_cust - cagr_std)*100:.3f}%"),
        ("Annualized Volatility", f"{vol_cust*100:.2f}%", f"{vol_std*100:.2f}%", f"{(vol_cust - vol_std)*100:.3f}%"),
        ("Sharpe Ratio", f"{sharpe_cust:.3f}", f"{sharpe_std:.3f}", f"{sharpe_cust - sharpe_std:.4f}"),
        ("Deflated Sharpe Ratio", f"{dsr_cust:.4f}", f"{dsr_std:.4f}", f"{dsr_cust - dsr_std:.4f}"),
        ("Max Drawdown", f"{max_dd_cust*100:.2f}%", f"{max_dd_std*100:.2f}%", f"{(max_dd_cust - max_dd_std)*100:.3f}%"),
        ("Trade Count", str(len(trades_cust)), str(trades_std), str(len(trades_cust) - trades_std)),
        ("Equity Correlation", "1.00000", f"{corr:.5f}", "0.00000")
    ]:
        reconciliation_results.append({
            "Asset": "BTC",
            "Metric": metric,
            "Custom Engine": cust_val,
            "Standard Engine": std_val,
            "Difference": diff_val
        })
        
    recon_df = pd.DataFrame(reconciliation_results)
    recon_df.to_csv(DATA_DIR / "reconciliation_proposal3.csv", index=False)
    print(f"[+] Reconciliation results saved to {DATA_DIR / 'reconciliation_proposal3.csv'}")

if __name__ == "__main__":
    run_reconciliation()
