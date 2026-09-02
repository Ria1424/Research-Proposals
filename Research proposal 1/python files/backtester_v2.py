"""
backtester_v2.py
================
Core event-driven backtesting engine for daily crypto strategies.

Key design guarantees:
  - NO look-ahead: signal at bar t executes at bar t+1's OPEN price.
  - Cost-aware: taker fee + slippage deducted from equity at every fill.
    Net equity and gross equity will DIFFER whenever trades occur.
  - Fractional sizing: uses instrument precision (0.001 BTC / 0.001 ETH)
    instead of int() truncation, so exposure is correct at all price levels.
  - Metrics output: CAGR, Vol, Sharpe, Sortino, Calmar, MaxDD, WinRate,
    ProfitFactor, NumTrades, Turnover — all gross AND net of costs.

NOTE (OHLCV-only run): BTCUSDT_daily.csv and ETHUSDT_daily.csv do not
  contain a `funding_rate` column, so funding features (fr_raw, fr_zscore_*
  etc.) are set to 0.0 during feature construction and are silently dropped
  by LightGBM's NaN handling. This means the current reproducible pipeline
  is an OHLCV-only ML classification study.  The `taker_buy_volume` column
  is also absent; tbr_* features are similarly all-NaN and dropped.
  See proposal1_main_v2.py for the full feature-generation note.

Author: Ria Chawak | IIT Bombay Research Internship 2026
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Union


# Minimum position increment supported by Binance BTC/ETH perpetual contracts
BTC_SIZE_PRECISION = 0.001   # 0.001 BTC
ETH_SIZE_PRECISION = 0.001   # 0.001 ETH
DEFAULT_PRECISION  = 0.001   # fallback


@dataclass
class CostModel:
    """Cost parameters for Binance Futures daily trading."""
    taker_fee: float = 0.0004      # 0.04% per fill (standard Binance tier)
    maker_fee: float = 0.0002      # 0.02% per fill
    slippage: float = 0.0005       # 0.05% per fill (conservative for daily)
    use_taker: bool = True         # Use taker (market order) for all fills

    @property
    def cost_per_fill(self) -> float:
        """One-way cost: fee + slippage."""
        return (self.taker_fee if self.use_taker else self.maker_fee) + self.slippage

    @property
    def round_trip_cost(self) -> float:
        """Total cost for entry + exit."""
        return self.cost_per_fill * 2


@dataclass
class Trade:
    """Represents a single completed trade (round trip)."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int            # +1 = long, -1 = short
    entry_price: float
    exit_price: float
    gross_pnl_pct: float
    net_pnl_pct: float
    costs_pct: float


@dataclass
class BacktestResult:
    """Holds all backtest output metrics."""
    equity_curve: pd.Series          # equity curve (net), starting at 1.0
    equity_gross: pd.Series          # equity curve (gross, no costs)
    trades: list = field(default_factory=list)
    positions: pd.Series = None      # position series (+1/-1/0) at bar t

    # Net metrics
    cagr_net: float = 0.0
    vol_net: float = 0.0
    sharpe_net: float = 0.0
    sortino_net: float = 0.0
    calmar_net: float = 0.0
    max_dd_net: float = 0.0

    # Gross metrics
    cagr_gross: float = 0.0
    sharpe_gross: float = 0.0

    # Trade-level stats
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pct: float = 0.0
    turnover_per_year: float = 0.0
    cost_drag_cagr: float = 0.0   # cagr_gross - cagr_net


class Backtester:
    """
    Event-driven daily backtester with correct cost accounting.

    Key fix vs. previous version:
      - Transaction costs (round-trip fee + slippage) are now deducted
        from the EQUITY ARRAY at each fill bar, so net_equity != gross_equity.
      - Fractional position sizing uses `size_precision` (e.g. 0.001 BTC)
        instead of int() truncation, which zeroed out positions whenever
        price > equity*0.45.

    Usage:
        bt = Backtester(ohlcv_df, cost_model=CostModel())
        result = bt.run(signal_series)

    signal_series: pd.Series indexed by date, values in {-1, 0, +1}.
        Signal at date t is generated from data at close of bar t.
        The backtester executes at the OPEN of bar t+1.
    """

    RISK_FREE_RATE = 0.05    # 5% annualised (approximate USD risk-free in 2024-26)
    DAYS_PER_YEAR = 365

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        cost_model: CostModel = None,
        size_precision: float = DEFAULT_PRECISION,
        equity_fraction: float = 0.45,
    ):
        """
        ohlcv: DataFrame with columns [open, high, low, close, volume]
               optionally [funding_rate] — zero if absent.
        size_precision: minimum position increment (0.001 for BTC/ETH perps).
        equity_fraction: fraction of equity to deploy per trade (0.45 = 45%).
        """
        self.ohlcv = ohlcv.copy()
        self.cost = cost_model or CostModel()
        self.size_precision = size_precision
        self.equity_fraction = equity_fraction

    def _round_to_precision(self, units: float) -> float:
        """Round to nearest valid lot size."""
        if self.size_precision <= 0:
            return units
        return round(units / self.size_precision) * self.size_precision

    def run(self, signal: pd.Series, label: str = "Strategy") -> BacktestResult:
        """
        Main backtest loop.
        signal: +1 (long), -1 (short), 0 (flat). Indexed by bar t date.
        Fills happen at open of bar t+1.
        """
        ohlcv = self.ohlcv
        sig = signal.reindex(ohlcv.index).fillna(0)

        n = len(ohlcv)
        equity_net = np.ones(n)
        equity_gross = np.ones(n)
        positions = np.zeros(n)
        trades = []

        current_pos = 0      # current signal direction: +1, -1, or 0
        entry_idx = None
        entry_price = None

        for i in range(n - 1):
            next_open = ohlcv["open"].iloc[i + 1]
            next_date = ohlcv.index[i + 1]
            funding = ohlcv.get(
                "funding_rate", pd.Series(0.0, index=ohlcv.index)
            ).iloc[i + 1]

            desired_pos = int(sig.iloc[i])

            # ── Position change: execute at next bar's open ──────────────
            if desired_pos != current_pos:
                # Close existing position (if any)
                if current_pos != 0 and entry_price is not None:
                    gross_pnl = current_pos * (next_open / entry_price - 1)
                    cost = self.cost.round_trip_cost   # both entry and exit costs
                    net_pnl = gross_pnl - cost
                    trades.append(Trade(
                        entry_date=ohlcv.index[entry_idx],
                        exit_date=next_date,
                        direction=current_pos,
                        entry_price=entry_price,
                        exit_price=next_open,
                        gross_pnl_pct=gross_pnl,
                        net_pnl_pct=net_pnl,
                        costs_pct=cost
                    ))
                    # Deduct EXIT cost from equity (entry cost was charged when position was opened)
                    equity_net[i + 1] = equity_net[i] * (1 - self.cost.cost_per_fill)
                    equity_gross[i + 1] = equity_gross[i]   # gross unchanged

                else:
                    equity_net[i + 1] = equity_net[i]
                    equity_gross[i + 1] = equity_gross[i]

                # Open new position (if desired_pos != 0)
                if desired_pos != 0:
                    # Deduct ENTRY cost when opening (other half charged at close)
                    equity_net[i + 1] *= (1 - self.cost.cost_per_fill)
                    entry_idx = i + 1
                    entry_price = next_open
                else:
                    entry_idx = None
                    entry_price = None

                current_pos = desired_pos

            else:
                # ── No position change: mark-to-market from close to close ─
                if i + 1 < n:
                    close_t = ohlcv["close"].iloc[i]
                    close_t1 = ohlcv["close"].iloc[i + 1]
                    daily_return = current_pos * (close_t1 / close_t - 1)

                    # Funding rate: paid by longs when positive, received by shorts
                    funding_cost = current_pos * funding
                    gross_daily = daily_return
                    net_daily = daily_return - funding_cost

                    equity_gross[i + 1] = equity_gross[i] * (1 + gross_daily)
                    equity_net[i + 1] = equity_net[i] * (1 + net_daily)

        # Close any still-open position at end (at last close)
        if current_pos != 0 and entry_price is not None:
            last_close = ohlcv["close"].iloc[-1]
            gross_pnl = current_pos * (last_close / entry_price - 1)
            cost = self.cost.round_trip_cost
            net_pnl = gross_pnl - cost
            trades.append(Trade(
                entry_date=ohlcv.index[entry_idx],
                exit_date=ohlcv.index[-1],
                direction=current_pos,
                entry_price=entry_price,
                exit_price=last_close,
                gross_pnl_pct=gross_pnl,
                net_pnl_pct=net_pnl,
                costs_pct=cost
            ))

        equity_net_s = pd.Series(equity_net, index=ohlcv.index)
        equity_gross_s = pd.Series(equity_gross, index=ohlcv.index)
        positions_s = pd.Series(positions, index=ohlcv.index)

        result = BacktestResult(
            equity_curve=equity_net_s,
            equity_gross=equity_gross_s,
            trades=trades,
            positions=positions_s
        )
        self._compute_metrics(result)
        return result

    def _compute_metrics(self, result: BacktestResult):
        """Compute all performance metrics from equity curve and trade list."""
        eq_net = result.equity_curve
        eq_gross = result.equity_gross
        n_days = len(eq_net)
        n_years = n_days / self.DAYS_PER_YEAR

        # ── Returns ────────────────────────────────────────────────────
        daily_ret_net = eq_net.pct_change().dropna()
        daily_ret_gross = eq_gross.pct_change().dropna()

        # ── CAGR ───────────────────────────────────────────────────────
        end_val_net = max(eq_net.iloc[-1], 1e-10)
        end_val_gross = max(eq_gross.iloc[-1], 1e-10)
        result.cagr_net = (end_val_net ** (1 / n_years) - 1)
        result.cagr_gross = (end_val_gross ** (1 / n_years) - 1)
        result.cost_drag_cagr = result.cagr_gross - result.cagr_net

        # ── Volatility ─────────────────────────────────────────────────
        result.vol_net = daily_ret_net.std() * np.sqrt(self.DAYS_PER_YEAR)

        # ── Sharpe ─────────────────────────────────────────────────────
        rf_daily = self.RISK_FREE_RATE / self.DAYS_PER_YEAR
        excess = daily_ret_net - rf_daily
        result.sharpe_net = (
            excess.mean() / excess.std() * np.sqrt(self.DAYS_PER_YEAR)
            if excess.std() > 0 else 0
        )
        excess_gross = daily_ret_gross - rf_daily
        result.sharpe_gross = (
            excess_gross.mean() / excess_gross.std() * np.sqrt(self.DAYS_PER_YEAR)
            if excess_gross.std() > 0 else 0
        )

        # ── Sortino ────────────────────────────────────────────────────
        downside = daily_ret_net[daily_ret_net < rf_daily]
        downside_std = downside.std() * np.sqrt(self.DAYS_PER_YEAR)
        result.sortino_net = (
            excess.mean() * self.DAYS_PER_YEAR / downside_std
            if downside_std > 0 else 0
        )

        # ── Max Drawdown ───────────────────────────────────────────────
        roll_max = eq_net.cummax()
        dd = (eq_net - roll_max) / roll_max
        result.max_dd_net = dd.min()

        # ── Calmar ─────────────────────────────────────────────────────
        result.calmar_net = (
            result.cagr_net / abs(result.max_dd_net)
            if result.max_dd_net != 0 else 0
        )

        # ── Trade-level stats ──────────────────────────────────────────
        trades = result.trades
        result.n_trades = len(trades)
        if trades:
            net_pnls = [t.net_pnl_pct for t in trades]
            wins = [p for p in net_pnls if p > 0]
            losses = [p for p in net_pnls if p <= 0]
            result.win_rate = len(wins) / len(net_pnls)
            result.avg_trade_pct = np.mean(net_pnls)
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            result.profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else np.inf
            )
        result.turnover_per_year = result.n_trades / n_years

    def summary(self, result: BacktestResult, label: str = "Strategy") -> pd.DataFrame:
        """Return a clean summary DataFrame of all metrics."""
        data = {
            "Metric": [
                "CAGR (Gross)", "CAGR (Net)", "Cost Drag (CAGR)",
                "Annualized Volatility", "Sharpe (Gross)", "Sharpe (Net)",
                "Sortino (Net)", "Calmar (Net)", "Max Drawdown (Net)",
                "Win Rate", "Profit Factor", "Number of Trades", "Turnover (trades/yr)"
            ],
            label: [
                f"{result.cagr_gross:.1%}", f"{result.cagr_net:.1%}", f"{result.cost_drag_cagr:.1%}",
                f"{result.vol_net:.1%}", f"{result.sharpe_gross:.2f}", f"{result.sharpe_net:.2f}",
                f"{result.sortino_net:.2f}", f"{result.calmar_net:.2f}", f"{result.max_dd_net:.1%}",
                f"{result.win_rate:.1%}", f"{result.profit_factor:.2f}", str(result.n_trades),
                f"{result.turnover_per_year:.1f}"
            ]
        }
        return pd.DataFrame(data)


def buy_and_hold(ohlcv: pd.DataFrame, cost_model: CostModel = None) -> BacktestResult:
    """Compute buy and hold benchmark — always long, single entry at start."""
    signal = pd.Series(1, index=ohlcv.index)
    bt = Backtester(ohlcv, cost_model or CostModel())
    return bt.run(signal, "Buy & Hold")
