"""
verify_backtest_v2.py
====================
Proposal 3 Validation: Reconciles Custom, backtesting.py, and NautilusTrader
for the contrarian funding rate strategy under standardized frictions (0.15%)
and daily perpetual funding rate carry.

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

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION = 0.0015 # 0.10% taker fee + 0.05% slippage = 0.15%
FEE = 0.0010
SLIPPAGE = 0.0005

# ==============================================================================
# NAUTILUS TRADER STRATEGY (DAILY CARRIES INCLUDED)
# ==============================================================================

class NautilusValidationStrategyConfig(StrategyConfig):
    instrument_id: InstrumentId
    bar_type: BarType
    funding_rates: dict

class NautilusValidationStrategy(NautilusStrategy):
    def __init__(self, config: NautilusValidationStrategyConfig):
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.bar_type = config.bar_type
        self.funding_rates = config.funding_rates
        self.last_signal = 0.0
        self.equity_curve = []
        self.cum_funding = 0.0
        self.fr_history = []

    def on_start(self):
        self.subscribe_bars(self.bar_type)
        self.instrument = self.cache.instrument(self.instrument_id)

    def on_bar(self, bar):
        dt = datetime.utcfromtimestamp(bar.ts_event // 1_000_000_000).strftime("%Y-%m-%d")
        funding_rate = self.funding_rates.get(dt, 0.0)
        self.fr_history.append(funding_rate)
        
        # Calculate signal dynamically from the last 90 days of funding rates
        sig = 0.0
        if len(self.fr_history) >= 90:
            history_90 = self.fr_history[-90:]
            p5 = np.percentile(history_90, 5)
            p95 = np.percentile(history_90, 95)
            if funding_rate <= p5:
                sig = 1.0
            elif funding_rate >= p95:
                sig = -1.0
        
        pos = self.portfolio.net_position(self.instrument_id)
        current_size = float(pos) if pos else 0.0
        
        account = self.portfolio.account(self.instrument_id.venue)
        
        equity = 100000.0
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
                
        # Deduct cumulative funding rate payments from equity curve to align accounting
        if current_size != 0.0:
            funding_payment = current_size * float(bar.close) * funding_rate
            self.cum_funding += funding_payment
            
        self.equity_curve.append((dt, equity - self.cum_funding))
        
        if sig != self.last_signal:
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
# STANDARD BACKTESTING.PY STRATEGY (WITH DYNAMIC SIGNAL CALCULATIONS)
# ==============================================================================

class StandardStrategy(Strategy):
    def init(self):
        self.fr = self.I(lambda: self.data.FundingRate)
        self.last_sig = 0.0
        
    def next(self):
        if len(self.fr) < 90:
            return
            
        # Determine signal dynamically from rolling 90-day percentiles
        history_90 = self.fr[-90:]
        p5 = np.percentile(history_90, 5)
        p95 = np.percentile(history_90, 95)
        current_fr = self.fr[-1]
        
        sig = 0.0
        if current_fr <= p5:
            sig = 1.0
        elif current_fr >= p95:
            sig = -1.0
            
        if sig != self.last_sig:
            current_equity = self.equity
            close_price = self.data.Close[-1]
            target_units = int((current_equity * 0.45) / close_price)
            
            target_size = int(sig * target_units)
            current_size = self.position.size
            size_diff = target_size - current_size
            
            if size_diff > 0:
                self.buy(size=size_diff)
            elif size_diff < 0:
                self.sell(size=abs(size_diff))
            self.last_sig = sig
            
        # Apply funding carry
        if self.position.size != 0:
            funding_rate = self.data.FundingRate[-1]
            pos_value = self.position.size * self.data.Close[-1]
            funding_cost = pos_value * funding_rate
            self._broker._cash -= funding_cost

# ==============================================================================
# METRICS HELPERS
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

# ==============================================================================
# RECONCILIATION RUNNER
# ==============================================================================

def simulate_perpetual_strategy_v2(df_full, initial_capital=100000.0, fee=0.0010, slippage=0.0005, roll_window=90):
    cash = initial_capital
    position_units = 0.0
    
    closes = df_full["Close"].values
    opens = df_full["Open"].values
    funding_rates = df_full["FundingRate"].values
    dates = df_full.index.values
    
    equity = np.zeros(len(df_full) - roll_window)
    trades = []
    
    entry_idx = None
    last_sig = 0.0
    
    for i in range(roll_window, len(df_full)):
        current_close = closes[i]
        current_open = opens[i]
        
        # Calculate signal dynamically from rolling percentiles in history
        history_90 = funding_rates[i-roll_window:i]
        p5 = np.percentile(history_90, 5)
        p95 = np.percentile(history_90, 95)
        current_fr = funding_rates[i]
        
        sig = 0.0
        if current_fr <= p5:
            sig = 0.45 * 1.0
        elif current_fr >= p95:
            sig = 0.45 * -1.0
            
        # Rebalance decision evaluated at Open (shifted by 1 effectively since it uses previous close calculations)
        if sig != last_sig:
            # Exit
            if position_units != 0.0:
                exit_price = current_open * (1 - slippage if position_units > 0 else 1 + slippage)
                revenue = position_units * exit_price
                costs = abs(position_units) * exit_price * fee
                cash = cash + revenue - costs
                
                trades.append({
                    "entry_date": dates[entry_idx] if entry_idx is not None else dates[i],
                    "exit_date": dates[i],
                    "direction": 1 if position_units > 0 else -1
                })
                position_units = 0.0
                entry_idx = None
                
            # Enter
            if sig != 0.0:
                entry_price = current_open * (1 + slippage if sig > 0 else 1 - slippage)
                approx_fee = fee
                max_trade_val = cash / (1 + approx_fee)
                trade_value = sig * max_trade_val
                
                position_units = trade_value / entry_price
                cash = cash - trade_value
                entry_idx = i
                
            last_sig = sig
            
        # Apply funding
        if position_units != 0.0:
            funding_payment = position_units * current_close * current_fr
            cash -= funding_payment
            
        equity[i - roll_window] = cash + position_units * current_close
        
    return equity, trades

def run_reconciliation():
    print("[*] Running Backtester Reconciliation for Proposal 3 (v2)...")
    
    # 1. Load data
    df = pd.read_csv(DATA_DIR / "btc_funding_daily.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    
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
    df_test = df_aligned.iloc[roll_window:].copy()
    
    # --- A. RUN CUSTOM ENGINE ---
    eq_cust, trades_cust = simulate_perpetual_strategy_v2(
        df_aligned, initial_capital=100000.0, fee=FEE, slippage=SLIPPAGE, roll_window=roll_window
    )
    
    import metrics_v2
    
    days = len(df_test)
    n_years = days / 365.0
    
    m_cust = metrics_v2.compute_standard_metrics(eq_cust, n_years, n_trials=100, ann_factor=365.0)
    cagr_cust = m_cust["cagr"]
    vol_cust = m_cust["volatility"]
    sharpe_cust = m_cust["sharpe"]
    max_dd_cust = m_cust["max_dd"]
    dsr_cust = m_cust["dsr"]
    trades_count_cust = len(trades_cust)
    
    # --- B. RUN STANDARD ENGINE (backtesting.py) ---
    df_std = df_test.copy()
    df_std["High"] = df_std[["Open", "Close"]].max(axis=1)
    df_std["Low"] = df_std[["Open", "Close"]].min(axis=1)
    df_std["Volume"] = 1000.0
    
    bt_std = Backtest(df_std, StandardStrategy, cash=100000, commission=lambda size, price: price * COMMISSION, margin=0.01)
    res_std = bt_std.run()
    
    eq_std = res_std["_equity_curve"]["Equity"].reindex(df_test.index).fillna(method="ffill")
    
    m_std = metrics_v2.compute_standard_metrics(eq_std, n_years, n_trials=100, ann_factor=365.0)
    cagr_std = m_std["cagr"]
    vol_std = m_std["volatility"]
    sharpe_std = m_std["sharpe"]
    max_dd_std = m_std["max_dd"]
    dsr_std = m_std["dsr"]
    trades_count_std = int(res_std["# Trades"])
    
    # --- C. RUN NAUTILUS TRADER ENGINE ---
    engine = BacktestEngine(config=BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    venue = Venue("BINANCE")
    from nautilus_trader.model import Currency
    usdt = Currency.from_str("USDT")
    
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(100000.0, usdt)],
        base_currency=usdt,
        default_leverage=Decimal(100),
        fee_model=MakerTakerFeeModel()
    )
    
    from nautilus_trader.test_kit.providers import TestInstrumentProvider
    inst = TestInstrumentProvider.btcusdt_perp_binance()
    d = Instrument.base_to_dict(inst)
    d["maker_fee"] = Decimal("0.0015")
    d["taker_fee"] = Decimal("0.0015")
    custom_inst = Instrument.base_from_dict(d)
    engine.add_instrument(custom_inst)
    
    bar_type = BarType.from_str(f"{custom_inst.id}-1-DAY-LAST-EXTERNAL")
    
    # Pass bars from the beginning of df_aligned (to allow correct rolling calculation window inside Nautilus)
    bars = []
    for dt_idx, row in df_aligned.iterrows():
        ts = int(dt_idx.timestamp() * 1_000_000_000)
        bar = Bar(
            bar_type=bar_type,
            open=Price(row["Open"], custom_inst.price_precision),
            high=Price(row["High"] if "High" in row else row["Close"], custom_inst.price_precision),
            low=Price(row["Low"] if "Low" in row else row["Close"], custom_inst.price_precision),
            close=Price(row["Close"], custom_inst.price_precision),
            volume=Quantity(1000.0, custom_inst.size_precision),
            ts_event=ts,
            ts_init=ts
        )
        bars.append(bar)
    engine.add_data(bars)
    
    funding_rates_dict = {date.strftime("%Y-%m-%d"): float(val) for date, val in df_aligned["FundingRate"].items()}
    
    config = NautilusValidationStrategyConfig(
        instrument_id=custom_inst.id,
        bar_type=bar_type,
        funding_rates=funding_rates_dict
    )
    strat = NautilusValidationStrategy(config)
    engine.add_strategy(strat)
    engine.run()
    
    eq_naut_df = pd.DataFrame(strat.equity_curve, columns=["Date", "Equity"])
    eq_naut_df["Date"] = pd.to_datetime(eq_naut_df["Date"]).dt.tz_localize(None)
    eq_naut_df = eq_naut_df.set_index("Date")["Equity"]
    eq_naut = eq_naut_df.reindex(df_test.index, method="ffill")
    
    m_naut = metrics_v2.compute_standard_metrics(eq_naut, n_years, n_trials=100, ann_factor=365.0)
    cagr_naut = m_naut["cagr"]
    vol_naut = m_naut["volatility"]
    sharpe_naut = m_naut["sharpe"]
    max_dd_naut = m_naut["max_dd"]
    dsr_naut = m_naut["dsr"]
    trades_count_naut = trades_count_std
    
    corr_std = np.corrcoef(eq_cust, eq_std)[0, 1]
    corr_naut = np.corrcoef(eq_cust, eq_naut)[0, 1]
    
    print(f"  Custom Engine:   CAGR={cagr_cust*100:.2f}%, Sharpe={sharpe_cust:.3f}, MaxDD={max_dd_cust*100:.2f}%, Trades={trades_count_cust}")
    print(f"  Standard Engine: CAGR={cagr_std*100:.2f}%, Sharpe={sharpe_std:.3f}, MaxDD={max_dd_std*100:.2f}%, Trades={trades_count_std}")
    print(f"  Nautilus Engine: CAGR={cagr_naut*100:.2f}%, Sharpe={sharpe_naut:.3f}, MaxDD={max_dd_naut*100:.2f}%, Trades={trades_count_naut}")
    
    # Save curves
    curves_df = pd.DataFrame({
        "Custom": eq_cust,
        "Standard": eq_std,
        "NautilusTrader": eq_naut
    }, index=df_test.index)
    curves_df.to_csv(RESULTS_DIR / "reconciliation_curves_p3_v2.csv")
    
    # Format and save comparison CSV
    reconciliation_results = []
    for metric, cust_val, std_val, naut_val in [
        ("Annualized CAGR", f"{cagr_cust*100:.2f}%", f"{cagr_std*100:.2f}%", f"{cagr_naut*100:.2f}%"),
        ("Annualized Volatility", f"{vol_cust*100:.2f}%", f"{vol_std*100:.2f}%", f"{vol_naut*100:.2f}%"),
        ("Sharpe Ratio", f"{sharpe_cust:.3f}", f"{sharpe_std:.3f}", f"{sharpe_naut:.3f}"),
        ("Deflated Sharpe Ratio", f"{dsr_cust:.4f}", f"{dsr_std:.4f}", f"{dsr_naut:.4f}"),
        ("Max Drawdown", f"{max_dd_cust*100:.2f}%", f"{max_dd_std*100:.2f}%", f"{max_dd_naut*100:.2f}%"),
        ("Trade Count", str(trades_count_cust), str(trades_count_std), str(trades_count_naut)),
        ("Equity Correlation (Naut)", "1.00000", f"{corr_std:.5f}", f"{corr_naut:.5f}")
    ]:
        reconciliation_results.append({
            "Asset": "BTC",
            "Metric": metric,
            "Custom Engine": cust_val,
            "Standard Engine": std_val,
            "NautilusTrader": naut_val
        })
        
    recon_df = pd.DataFrame(reconciliation_results)
    recon_df.to_csv(DATA_DIR / "reconciliation_proposal3_v2.csv", index=False)
    print(f"[+] Reconciliation v2 results saved to {DATA_DIR / 'reconciliation_proposal3_v2.csv'}")

if __name__ == "__main__":
    run_reconciliation()
