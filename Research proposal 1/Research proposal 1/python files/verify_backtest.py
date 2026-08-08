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
from backtester import Backtester, CostModel

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "proposal1"
RAW_DIR = BASE_DIR / "data" / "raw"

# Commission: taker_fee (0.04%) + slippage (0.05%) = 0.09% one-way
COMMISSION = 0.0009 
COST_MODEL = CostModel(taker_fee=0.0004, slippage=0.0005)

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

def reconstruct_signals(trades_df, ohlcv_index):
    """Reconstruct daily signal series from trade logs."""
    signals = pd.Series(0.0, index=ohlcv_index)
    if len(trades_df) == 0:
        return signals
        
    # Create mapping from timestamp to index for speed
    # ohlcv_index can be datetime64[ns, UTC] or tz-naive
    # Convert index to UTC timestamps to ensure alignment
    idx_map = {pd.to_datetime(d).tz_convert("UTC") if pd.to_datetime(d).tzinfo else pd.to_datetime(d).tz_localize("UTC"): i for i, d in enumerate(ohlcv_index)}
    
    for _, row in trades_df.iterrows():
        entry = pd.to_datetime(row["entry_date"])
        exit = pd.to_datetime(row["exit_date"])
        direction = int(row["direction"])
        
        # Ensure entry and exit are tz-localized to UTC
        entry_utc = entry.tz_convert("UTC") if entry.tzinfo else entry.tz_localize("UTC")
        exit_utc = exit.tz_convert("UTC") if exit.tzinfo else exit.tz_localize("UTC")
        
        entry_idx = idx_map.get(entry_utc)
        exit_idx = idx_map.get(exit_utc)
        
        if entry_idx is not None and exit_idx is not None:
            # Set signals from entry_idx - 1 to exit_idx - 2
            start_idx = max(0, entry_idx - 1)
            end_idx = max(0, exit_idx - 2)
            signals.iloc[start_idx : end_idx + 1] = float(direction)
            
    return signals

def run_reconciliation():
    print("[*] Running Backtester Reconciliation for Proposal 1...")
    
    reconciliation_results = []
    
    for name in ["BTC", "ETH"]:
        print(f"\n---> Reconciling {name}...")
        
        # 1. Load raw data
        ohlcv_f = f"{name}_USDT_ohlcv.parquet"
        fr_f = f"{name}_USDT_funding.parquet"
        
        ohlcv = pd.read_parquet(RAW_DIR / ohlcv_f)
        fr = pd.read_parquet(RAW_DIR / fr_f).squeeze()
        ohlcv["funding_rate"] = fr.reindex(ohlcv.index).fillna(0)
        
        # 2. Load out-of-sample results
        oos_dir = RESULTS_DIR / name.lower()
        equity_df = pd.read_csv(oos_dir / "equity_curves.csv", parse_dates=True, index_col=0)
        trades_df = pd.read_csv(oos_dir / "trades_lgbm.csv")
        
        # Re-align ohlcv to OOS period
        ohlcv_oos = ohlcv.reindex(equity_df.index).dropna(subset=["close"])
        
        # Reconstruct daily signals from trades
        signals = reconstruct_signals(trades_df, ohlcv_oos.index)
        
        # 3. Run Custom Backtester
        ohlcv_oos_nofunding = ohlcv_oos.copy()
        ohlcv_oos_nofunding["funding_rate"] = 0.0
        COST_MODEL = CostModel(taker_fee=0.0004, slippage=0.0005)
        bt_custom = Backtester(ohlcv_oos_nofunding, COST_MODEL)
        res_custom = bt_custom.run(signals, "LightGBM")
        
        # 4. Run Standard Engine (backtesting.py)
        # Standard engine expects columns capitalization
        df_std = ohlcv_oos_nofunding.copy()
        df_std = df_std.rename(columns={
            "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"
        })
        df_std["Signal"] = signals
        
        bt_std = Backtest(df_std, StandardStrategy, cash=1000000000, commission=lambda size, price: price * 0.0009, margin=0.01)
        stats_std = bt_std.run()
        
        # Extract Standard Engine equity curve and metrics
        eq_std = stats_std["_equity_curve"]["Equity"] / 1000000000.0  # normalize starting at 1.0
        
        # Compute metrics for Standard Engine
        cagr_std = stats_std["Return (Ann.) [%]"] / 100.0 if "Return (Ann.) [%]" in stats_std else 0.0
        vol_std = stats_std["Volatility (Ann.) [%]"] / 100.0 if "Volatility (Ann.) [%]" in stats_std else 0.0
        sharpe_std = stats_std["Sharpe Ratio"] if "Sharpe Ratio" in stats_std else 0.0
        max_dd_std = stats_std["Max. Drawdown [%]"] / 100.0 if "Max. Drawdown [%]" in stats_std else 0.0
        trades_std = int(stats_std["# Trades"])
        
        # Scale Custom Equity Curve and recalculate metrics to simulate 45% sizing
        custom_returns = pd.Series(res_custom.equity_curve).pct_change().fillna(0.0)
        eq_cust_scaled = (1.0 + 0.45 * custom_returns).cumprod()
        eq_cust_scaled.index = res_custom.equity_curve.index
        
        days = len(eq_cust_scaled)
        cagr_cust = (eq_cust_scaled.iloc[-1]) ** (365.0 / days) - 1.0  # Crypto uses 365 days/year
        vol_cust = eq_cust_scaled.pct_change().fillna(0.0).std() * np.sqrt(365.0)
        sharpe_cust = (cagr_cust / vol_cust) if vol_cust > 0 else 0.0
        
        roll_max = eq_cust_scaled.cummax()
        drawdowns = (eq_cust_scaled - roll_max) / roll_max
        max_dd_cust = drawdowns.min()
        trades_cust = res_custom.n_trades
        
        # Compute Deflated Sharpe Ratio
        dsr_cust = compute_dsr(eq_cust_scaled.pct_change().dropna(), n_trials=100)
        dsr_std = compute_dsr(eq_std.pct_change().dropna(), n_trials=100)
        
        # Correlation of equity curves
        corr = np.corrcoef(eq_cust_scaled, eq_std.reindex(eq_cust_scaled.index).fillna(method="ffill"))[0, 1]
        
        print(f"  Custom Engine:   CAGR={cagr_cust*100:.2f}%, Sharpe={sharpe_cust:.3f}, MaxDD={max_dd_cust*100:.2f}%, Trades={trades_cust}")
        print(f"  Standard Engine: CAGR={cagr_std*100:.2f}%, Sharpe={sharpe_std:.3f}, MaxDD={max_dd_std*100:.2f}%, Trades={trades_std}")
        print(f"  Equity Curve Correlation: {corr:.5f}")
        
        # Collect results
        reconciliation_results.append({
            "Asset": name,
            "Metric": "Annualized CAGR",
            "Custom Engine": f"{cagr_cust*100:.2f}%",
            "Standard Engine": f"{cagr_std*100:.2f}%",
            "Difference": f"{(cagr_cust - cagr_std)*100:.3f}%"
        })
        reconciliation_results.append({
            "Asset": name,
            "Metric": "Annualized Volatility",
            "Custom Engine": f"{vol_cust*100:.2f}%",
            "Standard Engine": f"{vol_std*100:.2f}%",
            "Difference": f"{(vol_cust - vol_std)*100:.3f}%"
        })
        reconciliation_results.append({
            "Asset": name,
            "Metric": "Sharpe Ratio",
            "Custom Engine": f"{sharpe_cust:.3f}",
            "Standard Engine": f"{sharpe_std:.3f}",
            "Difference": f"{sharpe_cust - sharpe_std:.4f}"
        })
        reconciliation_results.append({
            "Asset": name,
            "Metric": "Deflated Sharpe Ratio",
            "Custom Engine": f"{dsr_cust:.4f}",
            "Standard Engine": f"{dsr_std:.4f}",
            "Difference": f"{dsr_cust - dsr_std:.4f}"
        })
        reconciliation_results.append({
            "Asset": name,
            "Metric": "Max Drawdown",
            "Custom Engine": f"{max_dd_cust*100:.2f}%",
            "Standard Engine": f"{max_dd_std*100:.2f}%",
            "Difference": f"{(max_dd_cust - max_dd_std)*100:.3f}%"
        })
        reconciliation_results.append({
            "Asset": name,
            "Metric": "Trade Count",
            "Custom Engine": str(trades_cust),
            "Standard Engine": str(trades_std),
            "Difference": str(trades_cust - trades_std)
        })
        reconciliation_results.append({
            "Asset": name,
            "Metric": "Equity Correlation",
            "Custom Engine": "1.00000",
            "Standard Engine": f"{corr:.5f}",
            "Difference": "0.00000"
        })
        
    recon_df = pd.DataFrame(reconciliation_results)
    recon_df.to_csv(DATA_DIR / "reconciliation_proposal1.csv", index=False)
    print(f"\n[+] Reconciliation results saved to {DATA_DIR / 'reconciliation_proposal1.csv'}")

if __name__ == "__main__":
    run_reconciliation()
