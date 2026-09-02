# Quantitative Backtesting Engines Cross-Verification & Reconciliation Repository

This repository contains the code, raw datasets, walk-forward results, and research reports for three cryptocurrency quantitative trading strategies. The core engineering objective of this work is the cross-verification and mathematical reconciliation of a **Custom Event-Driven Vector Backtesting Engine** against an industry-accepted event-driven framework (**`backtesting.py`** in Python).

To ensure direct mathematical equivalence and isolate execution differences, all engines are aligned on:
1. **Position Sizing**: Fractional sizing at 45% of current equity, rounded to the minimum instrument lot size (0.001 BTC / 0.001 ETH on Binance perpetual contracts). Earlier integer-truncated sizing (`int()`) is corrected in `_v2` scripts.
2. **Execution Timing**: Next-bar Open fills (signals generated at close of day $t$ execute at the Open price of day $t+1$).
3. **Trading Costs**: Per-unit callable commission wrappers: `lambda size, price: abs(size) * price * commission_rate`. Costs are deducted from the equity curve at every fill (net equity ≠ gross equity when trades occur).
4. **Leverage & Margin**: Set leverage margin to 0.01 (100x leverage limit) in `backtesting.py` to satisfy position-flipping cash checks and prevent order cancellations.
5. **Mark-to-Market Equity**: NautilusTrader engine uses unrealised P&L = `cash_balance + position_size × current_close`, not cash-only staircase balance.

---

## 📁 Repository Directory Layout

Each research proposal folder is cleanly structured into subfolders for `data`, `results`, and `python files`, along with its respective research report document:

```
github/
│
├── Research proposal 1/                  # ML Classifier (LightGBM/RF/LR) on Perpetual Futures — OHLCV-only run
│   ├── data/                             # Raw price CSVs and reconciliation summaries (incl. reconciliation_proposal1_v2.csv)
│   ├── results/                          # Walk-forward trades, equity curves, and performance plots
│   ├── python files/
│   │   ├── backtester_v2.py              # [v2] Fixed cost path (net≠gross), fractional sizing
│   │   ├── backtester.py                 # Original backtester (retained for reference)
│   │   ├── features.py                   # [v2 fix] upper_shadow_pct corrected; OHLCV-only note
│   │   ├── proposal1_main_v2.py          # Walk-forward ML pipeline (OHLCV-only)
│   │   ├── verify_backtest_v2.py         # [v2 fixes] Fractional sizing, correct commission, MTM equity
│   │   ├── test_backtester_costs.py      # [NEW] Unit tests: zero-signal & flat-price cost path
│   │   ├── metrics_v2.py                 # Centralized metrics (shared across proposals)
│   │   └── ...other scripts
│   ├── pyproject.toml                    # Poetry environment configuration for Proposal 1
│   ├── poetry.lock                       # Locked dependency tree for Proposal 1
│   ├── Research_Proposal_1_Paper.docx    # Original double-column IEEE paper
│   ├── Research_Proposal_1_Paper_v2.docx # Reconciled IEEE Paper (v2) - formatting preserved, values updated
│   ├── Reserach Proposal 1 Report.docx   # Original single-column detailed report
│   └── Reserach Proposal 1 Report_v2.docx # Reconciled Detailed Report (v2) - formatting preserved, values updated
│
├── Research proposal 2/                  # LSTM vs. Classical Spread Pairs Trading
│   ├── data/                             # Cointegrated prices, model predictions, and reconciliation (incl. reconciliation_proposal2_v2.csv)
│   ├── results/                          # Metrics summary and performance plots
│   ├── python files/
│   │   ├── verify_backtest_v2.py         # [v2 fixes] Fractional sizing, correct commission, MTM equity
│   │   ├── metrics_v2.py                 # Centralized metrics (shared across proposals)
│   │   └── ...other scripts
│   ├── pyproject.toml                    # Poetry environment configuration for Proposal 2
│   ├── poetry.lock                       # Locked dependency tree for Proposal 2
│   ├── Research_Proposal_2_Paper.docx    # Original double-column IEEE paper
│   ├── Research_Proposal_2_Paper_v2.docx # Reconciled IEEE Paper (v2) - formatting preserved, values updated
│   ├── Research Proposal 2 Report.docx   # Original single-column detailed report
│   └── Research Proposal 2 Report_v2.docx # Reconciled Detailed Report (v2) - formatting preserved, values updated
│
├── Research proposal 3/                  # Contrarian perpetual futures funding rate Strategy
│   ├── data/                             # Price history, daily funding rates, and reconciliation (incl. reconciliation_proposal3_v2.csv)
│   ├── results/                          # Metrics summary, event study statistics, decay results
│   ├── python files/
│   │   ├── decay_study.py                # [NEW] Robust statistics (HAC, Block Bootstrap, Chow break test) & era decay
│   │   ├── verify_backtest_v2.py         # [v2 fixes] Fractional sizing, correct commission, MTM equity, measured trade count
│   │   ├── metrics_v2.py                 # Centralized metrics (shared across proposals)
│   │   └── ...other scripts
│   ├── pyproject.toml                    # Poetry environment configuration for Proposal 3
│   ├── poetry.lock                       # Locked dependency tree for Proposal 3
│   ├── Research_Proposal_3_Paper.docx    # Original double-column IEEE paper
│   ├── Research_Proposal_3_Paper_v2.docx # Reconciled IEEE Paper (v2) - formatting preserved, values updated
│   ├── Research Proposal 3 Report.docx   # Original single-column detailed report
│   └── Research Proposal 3 Report_v2.docx # Reconciled Detailed Report (v2) - formatting preserved, values updated
│
├── Proposal_Improvement_Deck.md           # [NEW] Priority-ranked presentation deck & improvement roadmap
├── Verification Report.docx              # Master comparison report (original)
├── Verification_Report_v2.docx           # Master Comparison Report (v2)
├── generate_reports_v2.py                # Automated generator script for Word reports
├── update_all_reports_v2.py              # Modifies only results tables inside original copied Word reports
├── metrics_v2.py                         # Centralized quantitative metrics library
├── Project_Review_31-08-2026.md          # Professor review notes
├── pyproject.toml                        # Global poetry workspace configuration
├── poetry.lock                           # Global locked dependency tree
└── README.md                             # Reproducibility documentation and mathematical specifications
```

---

## 🚀 Reproducibility Instructions

To replicate the reconciliation results and verify the mathematical equivalence between the custom and standard backtesting engines:

1. **Prerequisites**: Ensure you have Python 3.8+ installed along with the required libraries:
   ```bash
   pip install pandas numpy scipy backtesting docx python-docx lightgbm scikit-learn Keras tensorflow
   ```

2. **Numpy Compatibility Patch**:
   Standard `backtesting.py` uses deprecated numpy variables (like `np.bool8`) which were removed in Numpy 1.24+. All our verification scripts automatically apply a monkeypatch at start-up:
   ```python
   import numpy as np
   if not hasattr(np, "bool8"):
       np.bool8 = np.bool_
   ```

3. **Running Verification (Custom + backtesting.py + NautilusTrader)**:

   **Step 1 — Unit tests** (run once to confirm cost-path is correct):
   ```bash
   cd "Research proposal 1/python files"
   python test_backtester_costs.py
   ```
   Expected output: `ALL TESTS PASSED`

   **Step 2 — Proposal 1** (ML classifier, OHLCV-only):
   ```bash
   cd "Research proposal 1/python files"
   python proposal1_main_v2.py          # generates signals_btc_v2.csv and signals_eth_v2.csv
   python verify_backtest_v2.py         # runs all 3 engines, saves reconciliation_proposal1_v2.csv
   ```

   **Step 3 — Proposal 2** (LSTM pairs trading):
   ```bash
   cd "Research proposal 2/python files"
   python lstm_pairs_model.py           # generates lstm_predictions.csv
   python verify_backtest_v2.py         # runs all 3 engines, saves reconciliation_proposal2_v2.csv
   ```

   **Step 4 — Proposal 3** (Funding rate contrarian):
   ```bash
   cd "Research proposal 3/python files"
   python verify_backtest_v2.py         # runs all 3 engines, saves reconciliation_proposal3_v2.csv
   ```

   > **NautilusTrader install note**: NautilusTrader requires Python 3.10+ and Rust toolchain.
   > Install via: `pip install nautilus_trader`
   > Full docs: https://nautilustrader.io/docs/

4. **Regenerating Word Reports** (after re-running verification scripts):
   ```bash
   cd "github/"            # root of repo
   python update_all_reports_v2.py
   ```

---

## ⚖️ Transaction Costs and Callable Commission Wrappers

All three verification engines use a **per-unit callable** commission function:
```python
# CORRECT: charges abs(size) units at price per unit × rate
commission = lambda size, price: abs(size) * price * commission_rate
```

The previous implementation (`lambda size, price: price * commission_rate`) charged only a single-unit fee regardless of order size — underpaying by a factor of `n_units`. This is now fixed in all `_v2` scripts.

* **Proposal 1**: 0.10% taker fee + 0.05% slippage = **0.15% one-way cost**.
* **Proposal 2**: 0.10% fee + 0.05% slippage = **0.15% one-way cost**.
* **Proposal 3**: 0.10% fee + 0.05% slippage = **0.15% one-way cost** (funding carry applied separately).

---

## ⚠️ Data Availability Note (Proposal 1)

The `BTCUSDT_daily.csv` and `ETHUSDT_daily.csv` files supplied with this repository do **not** include a `funding_rate` column or a `taker_buy_volume` column. As a result:

- Funding-rate derived features (`fr_raw`, `fr_zscore_*`, `fr_roc_*`, etc.) default to zero and are dropped during model training.
- Taker-buy-ratio features (`tbr_5`, `tbr_10`, `tbr_20`) are all-NaN and are dropped during model training.

The reproducible walk-forward pipeline is therefore an **OHLCV-only ML classification study**. Funding rate and taker-flow features would require supplementary data from the Binance FAPI `/fundingRate` and `/aggTrades` endpoints.

---

## 📈 Deflated Sharpe Ratio (DSR)

We incorporate the **Deflated Sharpe Ratio (DSR)** (Bailey and López de Prado, 2014) to adjust for multiple testing selection bias and non-normally distributed returns.

The DSR is the Probabilistic Sharpe Ratio evaluated against the expected maximum Sharpe ratio SR* among the N tested trials:

```
DSR = Z[ (SR_hat - SR*) * sqrt(n-1) / sqrt(1 - skew*SR_hat + (kurt-1)/4 * SR_hat^2) ]

SR* = std(SR) * [(1 - Euler_gamma) * z_inv(1 - 1/N) + Euler_gamma * z_inv(1 - 1/(N*e))]
```

Where:
- `Z` is the standard normal CDF.
- `SR_hat` is the observed daily Sharpe ratio.
- `n` is the track record length in days.
- `skew`, `kurt` are the skewness and excess kurtosis of daily returns.
- `Euler_gamma` = 0.5772 (Euler-Mascheroni constant).
- `N` is the number of strategy trials tested.

---

## 📊 Summary of Reconciliation Verdicts

> **Note**: Equity curve correlation is reported as a **diagnostic only**, not as an acceptance criterion. A flat equity curve achieves high correlation with another flat curve regardless of P&L. Acceptance criteria: |Delta Sharpe| < 0.05 and |Delta CAGR| < 0.5 percentage points across all three engines.

| Research Proposal | Strategy / Asset | Reconciliation Status |
| :--- | :--- | :--- |
| **Proposal 1 (LightGBM)** | BTC Perpetual Futures | Pending v2 re-run |
| **Proposal 1 (LightGBM)** | ETH Perpetual Futures | Pending v2 re-run |
| **Proposal 2 (Pairs)** | Classical spread trading | Pending v2 re-run |
| **Proposal 2 (Pairs)** | LSTM Pairs (Raw) | Pending v2 re-run |
| **Proposal 2 (Pairs)** | LSTM Pairs (Thresh=0.02) | Pending v2 re-run |
| **Proposal 3 (Funding)** | Contrarian Funding Strategy | Pending v2 re-run |

> Re-run `verify_backtest_v2.py` in each proposal folder with the v2 code fixes applied to populate this table.
