"""
test_backtester_costs.py
========================
Unit tests for backtester_v2.py cost-path correctness.

Two required tests (per professor's review):
  Test 1 — Zero-signal path:
    All signals are 0. Strategy is flat the entire period.
    Expected: final equity == 1.0 exactly (no P&L, no costs).

  Test 2 — Flat-price round-trip:
    Price is constant (no market moves). Strategy enters long, then exits.
    Expected: equity loss == round_trip_cost exactly, since there is zero
    gross P&L and only transaction costs apply.

Run with:
    python test_backtester_costs.py

Or with pytest:
    pytest test_backtester_costs.py -v

Author: Ria Chawak | IIT Bombay Research Internship 2026
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Allow importing from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from backtester_v2 import Backtester, CostModel


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def make_flat_ohlcv(price: float = 50000.0, n_days: int = 10) -> pd.DataFrame:
    """Build an OHLCV DataFrame with a perfectly flat price."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({
        "open":   price,
        "high":   price,
        "low":    price,
        "close":  price,
        "volume": 1000.0,
    }, index=dates)
    return df


# ─────────────────────────────────────────────────────────────
# TEST 1: All-zero signal → equity must be exactly 1.0
# ─────────────────────────────────────────────────────────────

def test_zero_signal_no_cost():
    """
    Strategy is flat (signal = 0) for the entire period.
    No trades occur → no costs → equity stays at 1.0.
    """
    ohlcv = make_flat_ohlcv(price=50000.0, n_days=20)
    cost_model = CostModel(taker_fee=0.0010, slippage=0.0005)

    signal = pd.Series(0.0, index=ohlcv.index)
    bt = Backtester(ohlcv, cost_model)
    result = bt.run(signal, "ZeroSignal")

    final_net = result.equity_curve.iloc[-1]
    final_gross = result.equity_gross.iloc[-1]

    assert result.n_trades == 0, f"Expected 0 trades, got {result.n_trades}"
    assert abs(final_net - 1.0) < 1e-10, (
        f"TEST 1 FAILED: final net equity = {final_net:.10f}, expected 1.0"
    )
    assert abs(final_gross - 1.0) < 1e-10, (
        f"TEST 1 FAILED: final gross equity = {final_gross:.10f}, expected 1.0"
    )
    print(f"[PASS] Test 1 — Zero-signal path: net equity = {final_net:.10f}")


# ─────────────────────────────────────────────────────────────
# TEST 2: Flat price + single round-trip → cost exactly applied
# ─────────────────────────────────────────────────────────────

def test_flat_price_cost_equals_round_trip():
    """
    Price is constant (no P&L from market moves).
    Signal: long for first half, then flat.
    One trade occurs (open long, then close on exit).

    With a flat price:
      gross_pnl = 0
      net_pnl   = -round_trip_cost

    Expected final net equity:
      1.0 × (1 - cost_per_fill)   [entry cost deducted at open]
      × (1 - cost_per_fill)        [exit cost deducted at close]
      = (1 - cost_per_fill)^2

    Because cost_per_fill = taker_fee + slippage = 0.0010 + 0.0005 = 0.0015,
    expected = (1 - 0.0015)^2 = 0.9985^2 = 0.99700225
    """
    ohlcv = make_flat_ohlcv(price=50000.0, n_days=10)
    cost_model = CostModel(taker_fee=0.0010, slippage=0.0005)

    # Signal: long for first 5 bars, then flat
    signal = pd.Series(0.0, index=ohlcv.index)
    signal.iloc[:5] = 1.0

    bt = Backtester(ohlcv, cost_model)
    result = bt.run(signal, "FlatPriceRoundTrip")

    final_net = result.equity_curve.iloc[-1]
    final_gross = result.equity_gross.iloc[-1]

    cpf = cost_model.cost_per_fill          # 0.0015
    expected_net = (1 - cpf) ** 2           # entry cost + exit cost

    assert result.n_trades == 1, (
        f"Expected 1 trade, got {result.n_trades}"
    )
    assert abs(final_gross - 1.0) < 1e-10, (
        f"TEST 2 FAILED: gross equity = {final_gross:.10f}, expected 1.0 "
        f"(flat price → zero gross P&L)"
    )
    assert abs(final_net - expected_net) < 1e-8, (
        f"TEST 2 FAILED:\n"
        f"  final net equity = {final_net:.10f}\n"
        f"  expected         = {expected_net:.10f}\n"
        f"  difference       = {abs(final_net - expected_net):.2e}\n"
        f"  cost_per_fill    = {cpf}"
    )
    print(
        f"[PASS] Test 2 — Flat-price round-trip:\n"
        f"        net equity   = {final_net:.10f}\n"
        f"        expected     = {expected_net:.10f}  [(1 - {cpf})^2]\n"
        f"        cost drag    = {1 - final_net:.6f}"
    )


# ─────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  backtester_v2.py  —  Cost Path Unit Tests")
    print("=" * 60)
    failures = 0

    for test_fn in [test_zero_signal_no_cost, test_flat_price_cost_equals_round_trip]:
        try:
            test_fn()
        except AssertionError as e:
            print(f"\n[FAIL] {test_fn.__name__}:\n  {e}\n")
            failures += 1
        except Exception as e:
            print(f"\n[ERROR] {test_fn.__name__}:\n  {type(e).__name__}: {e}\n")
            failures += 1

    print("=" * 60)
    if failures == 0:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {failures} TEST(S) FAILED")
    print("=" * 60)
    sys.exit(failures)
