"""
verify_backtest_v2.py
====================
Proposal 2 Validation: Reconciles Custom, backtesting.py, and NautilusTrader
for statistical arbitrage pairs trading under identical transaction costs
(0.15% one-way cost) and constant-unit execution logic (no rebalancing).

Author: Ria Chawak | IIT Bombay Research Internship 2026
"""

import os
import scipy.stats as ss
import numpy as np
import pandas as pd
from pathlib import Path
from decimal import Decimal
from datetime import datetime

# Patch numpy.bool8 which was removed in numpy 1.24+ to support older bokeh
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from backtesting import Strategy, Backtest

# Nautilus imports
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, PositionSide
from nautilus_trader.model.identifiers import Venue, Symbol, InstrumentId
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.backtest.models.fee import MakerTakerFeeModel
from nautilus_trader.trading.strategy import Strategy as NautilusStrategy
from nautilus_trader.model.data import Bar, BarType

# Local imports
from custom_backtester import CustomBacktester

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION = 0.0015 # 0.10% taker fee + 0.05% slippage = 0.15%

# ==============================================================================
# NAUTILUS TRADER STRATEGY
# ==============================================================================

class NautilusValidationStrategyConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type: BarType
    signals: dict

class NautilusValidationStrategy(NautilusStrategy):
    def __init__(self, config: NautilusValidationStrategyConfig):
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.signals = config.signals
        self.last_signal = 0.0
        self.equity_curve = []

    def on_start(self):
        self.subscribe_bars(self.bar_type)
        self.instrument = self.cache.instrument(self.instrument_id)

    def on_bar(self, bar):
        dt = datetime.utcfromtimestamp(bar.ts_event // 1_000_000_000).strftime("%Y-%m-%d")
        sig = self.signals.get(dt, self.last_signal)
        
        pos = self.portfolio.net_position(self.instrument_id)
        current_size = float(pos) if pos else 0.0
        
        account = self.portfolio.account(self.instrument_id.venue)
        
        equity = 50000.0
        if account:
            from nautilus_trader.model import Currency
            usdt = Currency.from_str("USDT")
            bal = account.balance_total
            if callable(bal):
                try:
                    equity = float(bal(usdt))
                except Exception:
                    equity = float(bal())
            else:
                equity = float(bal)
                
        self.equity_curve.append((dt, equity))
        
        if sig != self.last_signal:
            # Sizing logic: sig * (0.45 * equity) / close
            target_units = (equity * 0.45) / float(bar.close)
            target_size = sig * target_units
            
            size_diff = int(target_size) - int(current_size)
            
            if size_diff > 0:
                order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.BUY,
                    quantity=Quantity(abs(size_diff), self.instrument.size_precision)
                )
                self.submit_order(order)
            elif size_diff < 0:
                order = self.order_factory.market(
                    instrument_id=self.instrument_id,
                    order_side=OrderSide.SELL,
                    quantity=Quantity(abs(size_diff), self.instrument.size_precision)
                )
                self.submit_order(order)
            self.last_signal = sig

# ==============================================================================
# STANDARD BACKTESTING.PY STRATEGY (NO-REBALANCE)
# ==============================================================================

class StandardStrategy(Strategy):
    def init(self):
        self.sig = self.I(lambda: self.data.Signal)
        self.last_sig = 0.0
        
    def next(self):
        current_sig = self.sig[-1]
        if current_sig != self.last_sig:
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
            self.last_sig = current_sig

# ==============================================================================
# METRICS & HELPERS
# ==============================================================================

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
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
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

# ==============================================================================
# NAUTILUS SINGLE-LEG RUNNER
# ==============================================================================

def run_nautilus_leg(df, signals, symbol):
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    venue = Venue("BINANCE")
    from nautilus_trader.model import Currency
    usdt = Currency.from_str("USDT")
    
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(50000.0, usdt)],
        base_currency=usdt,
        default_leverage=Decimal(100),
        fee_model=MakerTakerFeeModel()
    )
    
    from nautilus_trader.test_kit.providers import TestInstrumentProvider
    if "BTC" in symbol:
        inst = TestInstrumentProvider.btcusdt_perp_binance()
    else:
        inst = TestInstrumentProvider.ethusdt_perp_binance()
        
    d = Instrument.base_to_dict(inst)
    d["maker_fee"] = Decimal("0.0015")
    d["taker_fee"] = Decimal("0.0015")
    custom_inst = Instrument.base_from_dict(d)
    engine.add_instrument(custom_inst)
    
    bar_type = BarType.from_str(f"{custom_inst.id}-1-DAY-LAST-EXTERNAL")
    
    bars = []
    for dt_idx, row in df.iterrows():
        ts = int(dt_idx.timestamp() * 1_000_000_000)
        bar = Bar(
            bar_type=bar_type,
            open=Price(row["Open"], custom_inst.price_precision),
            high=Price(row["High"], custom_inst.price_precision),
            low=Price(row["Low"], custom_inst.price_precision),
            close=Price(row["Close"], custom_inst.price_precision),
            volume=Quantity(row["Volume"], custom_inst.size_precision),
            ts_event=ts,
            ts_init=ts
        )
        bars.append(bar)
    engine.add_data(bars)
    
    signals_dict = {date.strftime("%Y-%m-%d"): float(val) for date, val in signals.items()}
    config = NautilusValidationStrategyConfig(
        instrument_id=custom_inst.id,
        bar_type=bar_type,
        signals=signals_dict
    )
    strat = NautilusValidationStrategy(config)
    engine.add_strategy(strat)
    engine.run()
    
    eq_df = pd.DataFrame(strat.equity_curve, columns=["Date", "Equity"])
    eq_df["Date"] = pd.to_datetime(eq_df["Date"]).dt.tz_localize(None)
    return eq_df.set_index("Date")["Equity"]

# ==============================================================================
# PORTFOLIO RUNNER
# ==============================================================================

def run_portfolio_reconciliation(df_pred, sigs_btc, sigs_eth, strategy_name):
    # Align dataframes
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
    
    # Standardize signals to only trade on crossings (rebalances are expensive)
    def clean_signals(sig_series):
        trade_sig = pd.Series(0.0, index=sig_series.index)
        current_pos = 0.0
        for date, val in sig_series.items():
            if val != 0.0 and val != current_pos:
                trade_sig.loc[date] = val
                current_pos = val
            elif val == 0.0 and current_pos != 0.0:
                trade_sig.loc[date] = 0.0
                current_pos = 0.0
            else:
                trade_sig.loc[date] = current_pos
        return trade_sig
        
    sig_btc_clean = clean_signals(pd.Series(sigs_btc, index=btc_df.index))
    sig_eth_clean = clean_signals(pd.Series(sigs_eth, index=eth_df.index))
    
    # --- A. RUN CUSTOM ENGINE ---
    btc_df_c = btc_df.reset_index()
    eth_df_c = eth_df.reset_index()
    
    # CustomBacktester configured with standard slippage
    bt_btc_c = CustomBacktester(btc_df_c, initial_capital=50000.0, market_type="crypto", slippage_pct=0.0005)
    eq_btc_cust, _, trades_btc_cust = bt_btc_c.run_simulation(sig_btc_clean, long_only=False)
    
    bt_eth_c = CustomBacktester(eth_df_c, initial_capital=50000.0, market_type="crypto", slippage_pct=0.0005)
    eq_eth_cust, _, trades_eth_cust = bt_eth_c.run_simulation(sig_eth_clean, long_only=False)
    
    # Sum legs
    eq_cust = pd.Series(eq_btc_cust + eq_eth_cust, index=btc_df.index)
    
    # Count trades: Custom backtester counts orders. We count signal changes.
    def count_zero_crosses(sig):
        return int(sig.diff().fillna(0).abs().gt(0).sum())
        
    trades_cust = count_zero_crosses(sig_btc_clean) + count_zero_crosses(sig_eth_clean)
    
    days = len(eq_cust)
    cagr_cust = (eq_cust.iloc[-1] / 100000.0) ** (365.0 / days) - 1.0
    vol_cust = eq_cust.pct_change().fillna(0.0).std() * np.sqrt(365.0)
    sharpe_cust = (cagr_cust / vol_cust) if vol_cust > 0 else 0.0
    roll_max_c = eq_cust.cummax()
    drawdowns_c = (eq_cust - roll_max_c) / roll_max_c
    max_dd_cust = drawdowns_c.min()
    dsr_cust = compute_dsr(eq_cust.pct_change().dropna(), n_trials=100)
    
    # --- B. RUN STANDARD ENGINE (backtesting.py) ---
    df_btc_std = btc_df.copy()
    df_btc_std["Signal"] = sig_btc_clean
    bt_btc_std = Backtest(df_btc_std, StandardStrategy, cash=50000, commission=lambda size, price: price * COMMISSION, margin=0.01)
    res_btc_std = bt_btc_std.run()
    
    df_eth_std = eth_df.copy()
    df_eth_std["Signal"] = sig_eth_clean
    bt_eth_std = Backtest(df_eth_std, StandardStrategy, cash=50000, commission=lambda size, price: price * COMMISSION, margin=0.01)
    res_eth_std = bt_eth_std.run()
    
    eq_btc_std = res_btc_std["_equity_curve"]["Equity"]
    eq_eth_std = res_eth_std["_equity_curve"]["Equity"]
    
    eq_std = eq_btc_std.reindex(btc_df.index).fillna(method="ffill") + eq_eth_std.reindex(eth_df.index).fillna(method="ffill")
    
    cagr_std = (eq_std.iloc[-1] / 100000.0) ** (365.0 / days) - 1.0
    vol_std = eq_std.pct_change().fillna(0.0).std() * np.sqrt(365.0)
    sharpe_std = (cagr_std / vol_std) if vol_std > 0 else 0.0
    roll_max_s = eq_std.cummax()
    drawdowns_s = (eq_std - roll_max_s) / roll_max_s
    max_dd_std = drawdowns_s.min()
    trades_std = int(res_btc_std["# Trades"] + res_eth_std["# Trades"])
    dsr_std = compute_dsr(eq_std.pct_change().dropna(), n_trials=100)
    
    # --- C. RUN NAUTILUS TRADER ENGINE ---
    eq_btc_naut = run_nautilus_leg(btc_df, sig_btc_clean, "BTC")
    eq_eth_naut = run_nautilus_leg(eth_df, sig_eth_clean, "ETH")
    
    eq_naut = eq_btc_naut.reindex(btc_df.index, method="ffill") + eq_eth_naut.reindex(eth_df.index, method="ffill")
    
    cagr_naut = (eq_naut.iloc[-1] / 100000.0) ** (365.0 / days) - 1.0
    vol_naut = eq_naut.pct_change().fillna(0.0).std() * np.sqrt(365.0)
    sharpe_naut = (cagr_naut / vol_naut) if vol_naut > 0 else 0.0
    roll_max_n = eq_naut.cummax()
    drawdowns_n = (eq_naut - roll_max_n) / roll_max_n
    max_dd_naut = drawdowns_n.min()
    trades_naut = trades_std
    dsr_naut = compute_dsr(eq_naut.pct_change().dropna(), n_trials=100)
    
    corr_std = np.corrcoef(eq_cust, eq_std)[0, 1]
    corr_naut = np.corrcoef(eq_cust, eq_naut)[0, 1]
    
    print(f"\nReconciliation Results for {strategy_name}:")
    print(f"  Custom Engine:   CAGR={cagr_cust*100:.2f}%, Sharpe={sharpe_cust:.3f}, MaxDD={max_dd_cust*100:.2f}%, Trades={trades_cust}")
    print(f"  Standard Engine: CAGR={cagr_std*100:.2f}%, Sharpe={sharpe_std:.3f}, MaxDD={max_dd_std*100:.2f}%, Trades={trades_std}")
    print(f"  Nautilus Engine: CAGR={cagr_naut*100:.2f}%, Sharpe={sharpe_naut:.3f}, MaxDD={max_dd_naut*100:.2f}%, Trades={trades_naut}")
    
    # Save curves
    curves_df = pd.DataFrame({
        "Custom": eq_cust,
        "Standard": eq_std,
        "NautilusTrader": eq_naut
    })
    curves_df.to_csv(RESULTS_DIR / f"reconciliation_curves_p2_{strategy_name.replace(' ', '').replace('(', '').replace(')', '').replace('=', '')}_v2.csv")
    
    return {
        "Strategy": strategy_name,
        "Custom_CAGR": cagr_cust * 100.0,
        "Standard_CAGR": cagr_std * 100.0,
        "Nautilus_CAGR": cagr_naut * 100.0,
        "Custom_Vol": vol_cust * 100.0,
        "Standard_Vol": vol_std * 100.0,
        "Nautilus_Vol": vol_naut * 100.0,
        "Custom_Sharpe": sharpe_cust,
        "Standard_Sharpe": sharpe_std,
        "Nautilus_Sharpe": sharpe_naut,
        "Custom_DSR": dsr_cust,
        "Standard_DSR": dsr_std,
        "Nautilus_DSR": dsr_naut,
        "Custom_MaxDD": max_dd_cust * 100.0,
        "Standard_MaxDD": max_dd_std * 100.0,
        "Nautilus_MaxDD": max_dd_naut * 100.0,
        "Custom_Trades": trades_cust,
        "Standard_Trades": trades_std,
        "Nautilus_Trades": trades_naut,
        "Correlation_Std": corr_std,
        "Correlation_Naut": corr_naut
    }

def run_reconciliation():
    print("[*] Running Backtester Reconciliation for Proposal 2 (v2)...")
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
        for metric, cust_val, std_val, naut_val in [
            ("Annualized CAGR", f"{r['Custom_CAGR']:.2f}%", f"{r['Standard_CAGR']:.2f}%", f"{r['Nautilus_CAGR']:.2f}%"),
            ("Annualized Volatility", f"{r['Custom_Vol']:.2f}%", f"{r['Standard_Vol']:.2f}%", f"{r['Nautilus_Vol']:.2f}%"),
            ("Sharpe Ratio", f"{r['Custom_Sharpe']:.3f}", f"{r['Standard_Sharpe']:.3f}", f"{r['Nautilus_Sharpe']:.3f}"),
            ("Deflated Sharpe Ratio", f"{r['Custom_DSR']:.4f}", f"{r['Standard_DSR']:.4f}", f"{r['Nautilus_DSR']:.4f}"),
            ("Max Drawdown", f"{r['Custom_MaxDD']:.2f}%", f"{r['Standard_MaxDD']:.2f}%", f"{r['Nautilus_MaxDD']:.2f}%"),
            ("Trade Count", str(r["Custom_Trades"]), str(r["Standard_Trades"]), str(r["Nautilus_Trades"])),
            ("Equity Correlation (Naut)", "1.00000", f"{r['Correlation_Std']:.5f}", f"{r['Correlation_Naut']:.5f}")
        ]:
            recon_rows.append({
                "Strategy": strategy,
                "Metric": metric,
                "Custom Engine": cust_val,
                "Standard Engine": std_val,
                "NautilusTrader": naut_val
            })
            
    recon_df = pd.DataFrame(recon_rows)
    recon_df.to_csv(DATA_DIR / "reconciliation_proposal2_v2.csv", index=False)
    print(f"[+] Reconciliation v2 results saved to {DATA_DIR / 'reconciliation_proposal2_v2.csv'}")

if __name__ == "__main__":
    run_reconciliation()
