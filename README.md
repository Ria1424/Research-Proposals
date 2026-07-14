# Quantitative Backtesting Engines Cross-Verification & Reconciliation Repository

This repository contains the code, raw datasets, walk-forward results, and research reports for three cryptocurrency quantitative trading strategies. The core engineering objective of this work is the cross-verification and mathematical reconciliation of a **Custom Event-Driven Vector Backtesting Engine** against an industry-accepted event-driven framework (**`backtesting.py`** in Python).

To ensure direct mathematical equivalence and isolate execution differences, both engines were aligned on:
1. **Position Sizing & Compounding**: Daily compounding and rebalancing using 45% of current equity (0.45 weight).
2. **Execution Timing**: Next-bar Open fills (signals generated at close of day $t$ execute at the Open price of day $t+1$).
3. **Trading Costs**: Fully fee and slippage inclusive (using custom callable commission wrappers in `backtesting.py` to bypass unit-price inflation bugs).
4. **Leverage & Margin**: Set leverage margin to 0.01 (100x leverage limit) in `backtesting.py` to satisfy position-flipping cash checks and prevent order cancellations.

---

## 📁 Repository Directory Layout

Each research proposal folder is cleanly structured into subfolders for `data`, `results`, and `python files`, along with its respective research report document:

```
github/
│
├── Research proposal 1/                 # ML Classifier (LightGBM) on Perpetual Futures
│   ├── data/                            # Raw price CSVs and reconciliation summaries
│   ├── results/                         # Walk-forward trades, equity curves, and feature importance
│   ├── python files/                    # Core pipeline and validation scripts
│   └── RiaChawak_Research Propsal1.docx # Reference Word report
│
├── Research proposal 2/                 # LSTM vs. Classical Spread Pairs Trading
│   ├── data/                            # Cointegrated prices, model predictions, and reconciliation
│   ├── results/                         # Metrics summary and performance plots
│   ├── python files/                    # Core deep learning and execution scripts
│   └── Research Proposal 2_RiaC.docx    # Reference Word report
│
├── Research proposal 3/                 # Contrarian perpetual futures funding rate Strategy
│   ├── data/                            # Price history, daily funding rates, and reconciliation
│   ├── results/                         # Metrics summary and event study statistics
│   ├── python files/                    # Event studies and execution scripts
│   └── Research Proposal3_RiaC.docx    # Reference Word report
│
├── Backtest_Verification_Report.docx    # Master comparison report (Custom vs. Standard Results)
└── README.md                            # Reproducibility documentation and mathematical specifications
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

3. **Running Verification**:
   Navigate to the `python files/` folder of any research proposal and execute `verify_backtest.py`. For example, for Proposal 1:
   ```bash
   cd "Research proposal 1/python files"
   python verify_backtest.py
   ```
   This will run both engines side-by-side, print the CAGR, Sharpe, DSR, Max Drawdown, and Correlation metrics, and save them in the `data/` folder as `reconciliation_proposalX.csv`.

---

## ⚖️ Transaction Costs and Callable Commission Wrappers

To run standard `backtesting.py` with transaction costs without triggering cash-check failures, we must bypass a known bug in its default float commission handler. 

When a float (e.g. `0.0015`) is passed to the `Backtest` constructor:
$$\text{adjusted\_price\_plus\_commission} = \text{price} \times (1 + \text{order\_size} \times \text{commission\_relative})$$
This incorrectly adds the **total order commission** to the **unit price**, multiplying execution costs by the order size and causing immediate cash exhaustion.

To resolve this, all our verification scripts pass a **callable lambda function** for `commission`, which returns the commission **per unit**:
```python
commission = lambda size, price: price * commission_ratio
```
This forces `backtesting.py` to evaluate commissions on a per-unit basis, permitting direct, fee-inclusive reconciliation:
* **Proposal 1**: 0.04% taker fee + 0.05% slippage = **0.09% one-way cost**.
* **Proposal 2**: 0.10% fee + 0.05% slippage = **0.15% one-way cost**.
* **Proposal 3**: Run fee-free for reconciliation (0.0% cost) because the custom perpetual simulator closes and reopens the entire position daily (paying full transaction fees daily), whereas standard libraries only pay fees on the daily rebalanced difference.

---

## 📈 Deflated Sharpe Ratio (DSR)

As requested, we have incorporated the **Deflated Sharpe Ratio (DSR)** (Bailey and López de Prado, 2014) to adjust for multiple testing selection bias and non-normally distributed returns.

The DSR is calculated as the Probabilistic Sharpe Ratio (PSR) evaluated against the expected maximum Sharpe ratio among the $N$ tested trials ($\widehat{SR}^*$):

$$DSR = \widehat{PSR}(\widehat{SR}^*) = Z \left[ \frac{(\widehat{SR} - \widehat{SR}^*) \sqrt{n-1}}{\sqrt{1 - \widehat{\gamma}_3 \widehat{SR} + \frac{\widehat{\gamma}_4 - 1}{4} \widehat{SR}^2}} \right]$$

Where:
* $Z$ is the cumulative distribution function (CDF) of the standard normal distribution.
* $\widehat{SR}$ is the observed daily Sharpe ratio.
* $n$ is the track record length in days.
* $\widehat{\gamma}_3$ and $\widehat{\gamma}_4$ are the skewness and kurtosis of the daily returns.
* $\widehat{SR}^*$ is the Expected Maximum Sharpe Ratio benchmark under the null hypothesis (calculated based on the number of trials $N$):
  $$\widehat{SR}^* = \text{std}(\widehat{SR}) \times \left( (1 - \gamma_e) Z^{-1}(1 - \frac{1}{N}) + \gamma_e Z^{-1}(1 - \frac{1}{N \cdot e}) \right)$$
  (where $\gamma_e \approx 0.5772$ is the Euler-Mascheroni constant).

---

## 📊 Summary of Reconciliation Verdicts

| Research Proposal | Strategy / Asset | Equity Correlation | Reconciliation Status |
| :--- | :--- | :---: | :---: |
| **Proposal 1 (LightGBM)** | BTC Perpetual Futures | **99.23%** | **Reconciled (Fee-Inclusive)** |
| **Proposal 1 (LightGBM)** | ETH Perpetual Futures | **96.61%** | **Reconciled (Fee-Inclusive)** |
| **Proposal 2 (Pairs)** | Classical spread trading | **98.25%** | **Reconciled (Fee-Inclusive)** |
| **Proposal 2 (Pairs)** | LSTM Pairs (Raw) | **99.56%** | **Reconciled (Fee-Inclusive)** |
| **Proposal 2 (Pairs)** | LSTM Pairs (Thresh=0.02) | **99.51%** | **Reconciled (Fee-Inclusive)** |
| **Proposal 3 (Funding)** | Contrarian Funding Strategy | **100.00%** | **Perfect Match (Fee-Isolated)** |
