"""
proposal1_main.py
=================
Proposal 1: LightGBM vs MA Crossover on BTC/ETH Daily Perpetual Futures

Full walk-forward experiment:
  - 365-day expanding training window
  - 90-day OOS test window
  - 5-day embargo between train and test
  - LightGBM with purged hyperparameter tuning inside each training fold
  - MA crossover baseline re-run on same OOS windows for fair comparison

Outputs (saved to results/proposal1/):
  - metrics_btc.csv / metrics_eth.csv        (per-fold and aggregate metrics)
  - equity_curve_btc.csv / equity_curve_eth.csv
  - feature_importance_btc.csv / feature_importance_eth.csv
  - trades_btc.csv / trades_eth.csv
  - comparison_table.csv                     (LGBM vs MA vs B&H)
  - summary_report.txt                       (human-readable summary)

Usage:
    python proposal1_main.py                   # runs on BTC and ETH
    python proposal1_main.py --instrument BTC  # single instrument

Author: Ria Chawak | IIT Bombay Research Internship 2026
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import ParameterGrid
from pathlib import Path
import warnings
import argparse
import json
from datetime import datetime

warnings.filterwarnings("ignore")

# Local imports
from backtester import Backtester, CostModel, buy_and_hold
from features import build_features

FEAT_DIR = Path("data/features")
RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results/proposal1")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COST_MODEL = CostModel(taker_fee=0.0004, slippage=0.0005)
RISK_FREE = 0.05

# -- Walk-Forward Parameters -----------------------------------------------
TRAIN_DAYS = 365      # expanding training window (minimum)
TEST_DAYS  = 90       # OOS test window per fold
EMBARGO    = 5        # days gap between train end and test start
PROBA_HIGH = 0.55     # threshold to go long
PROBA_LOW  = 0.45     # threshold to go short

# -- LightGBM Hyperparameter Search Space ---------------------------------
PARAM_GRID = {
    "n_estimators":    [200, 400],
    "learning_rate":   [0.02, 0.05],
    "num_leaves":      [20, 40],
    "min_child_samples": [20, 30],
    "feature_fraction": [0.7, 0.9],
    "bagging_fraction": [0.8, 1.0],
    "bagging_freq":    [1],
    "lambda_l1":       [0.1],
    "lambda_l2":       [0.1],
}

FIXED_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
}

INSTRUMENTS = {
    "BTC": ("BTC_USDT_ohlcv.parquet", "BTC_USDT_funding.parquet", "BTC_features.parquet"),
    "ETH": ("ETH_USDT_ohlcv.parquet", "ETH_USDT_funding.parquet", "ETH_features.parquet"),
}


# ==============================================================================
# LABELLING
# ==============================================================================

def make_binary_labels(close: pd.Series) -> pd.Series:
    """
    Label at bar t = 1 if close[t+1] > close[t], else 0.
    Uses shift(-1) so label at t depends on t+1 close.
    IMPORTANT: drop the last row (label is NaN since t+1 doesn't exist).
    """
    fwd_ret = close.pct_change().shift(-1)
    labels = (fwd_ret > 0).astype(int)
    return labels


def make_ma_signal(close: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """
    EMA fast / EMA slow crossover signal.
    +1 when fast EMA > slow EMA, -1 when fast EMA < slow EMA.
    Signal at bar t uses data up to and including bar t.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    sig = pd.Series(0, index=close.index)
    sig[ema_fast > ema_slow] = 1
    sig[ema_fast < ema_slow] = -1
    return sig


# ==============================================================================
# WALK-FORWARD ENGINE
# ==============================================================================

def purged_cv_tune(X_train: pd.DataFrame, y_train: pd.Series,
                   n_splits: int = 4) -> dict:
    """
    Purged time-series cross-validation for hyperparameter tuning.
    Splits the training window chronologically, with embargo between folds.
    Returns best params dict.
    """
    n = len(X_train)
    fold_size = n // (n_splits + 1)
    best_score = np.inf
    best_params = list(ParameterGrid(PARAM_GRID))[0]

    for params in ParameterGrid(PARAM_GRID):
        fold_scores = []
        for i in range(1, n_splits + 1):
            train_end = i * fold_size
            val_start = train_end + EMBARGO
            val_end = val_start + fold_size
            if val_end > n:
                break

            X_tr = X_train.iloc[:train_end]
            y_tr = y_train.iloc[:train_end]
            X_val = X_train.iloc[val_start:val_end]
            y_val = y_train.iloc[val_start:val_end]

            all_params = {**FIXED_PARAMS, **params}
            n_est = all_params.pop("n_estimators", 200)

            model = lgb.LGBMClassifier(**all_params, n_estimators=n_est,
                                        early_stopping_rounds=30, verbose=-1)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                 lgb.log_evaluation(-1)])
            val_pred = model.predict_proba(X_val)[:, 1]
            from sklearn.metrics import log_loss
            score = log_loss(y_val, val_pred)
            fold_scores.append(score)

        avg_score = np.mean(fold_scores) if fold_scores else np.inf
        if avg_score < best_score:
            best_score = avg_score
            best_params = params

    return best_params


def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> lgb.LGBMClassifier:
    """Train final LightGBM model on full training set with best params."""
    all_params = {**FIXED_PARAMS, **params}
    n_est = all_params.pop("n_estimators", 200)
    model = lgb.LGBMClassifier(**all_params, n_estimators=n_est, verbose=-1)
    model.fit(X_train, y_train)
    return model


def run_walk_forward(ohlcv: pd.DataFrame, features: pd.DataFrame,
                     name: str = "ASSET") -> dict:
    """
    Main walk-forward loop for Proposal 1.
    Returns dict with all results.
    """
    close = ohlcv["close"]
    labels = make_binary_labels(close)
    ma_signal = make_ma_signal(close, fast=50, slow=200)

    # Align features and labels (drop columns that are entirely NaN, then drop rows with NaNs)
    features_clean = features.dropna(axis=1, how='all')
    valid_idx = features_clean.dropna().index.intersection(labels.dropna().index)
    X = features_clean.loc[valid_idx]
    y = labels.loc[valid_idx]

    # The first TRAIN_DAYS rows are the initial training window
    first_test_start = TRAIN_DAYS

    all_oos_dates = []
    all_lgbm_signals = []
    all_ma_signals = []
    fold_metrics = []
    feat_importances = []

    fold_num = 0
    test_start_idx = first_test_start

    print(f"\n{'='*60}")
    print(f"Walk-Forward: {name} | Train>={TRAIN_DAYS}d | Test={TEST_DAYS}d | Embargo={EMBARGO}d")
    print(f"{'='*60}")

    while test_start_idx + TEST_DAYS <= len(X):
        # -- Define windows --------------------------------------------
        train_end_idx = test_start_idx - EMBARGO
        train_end_idx = max(train_end_idx, TRAIN_DAYS)
        test_end_idx = min(test_start_idx + TEST_DAYS, len(X))

        X_train = X.iloc[:train_end_idx]
        y_train = y.iloc[:train_end_idx]
        X_test  = X.iloc[test_start_idx:test_end_idx]
        y_test  = y.iloc[test_start_idx:test_end_idx]
        test_dates = X.index[test_start_idx:test_end_idx]

        train_start_date = X_train.index[0].date()
        train_end_date   = X_train.index[-1].date()
        test_start_date  = test_dates[0].date()
        test_end_date    = test_dates[-1].date()

        print(f"\nFold {fold_num+1}: Train [{train_start_date}->{train_end_date}] "
              f"({len(X_train)}d) | Test [{test_start_date}->{test_end_date}] ({len(X_test)}d)")

        # -- Tune hyperparameters on training window -------------------
        print(f"  Tuning hyperparameters on {len(ParameterGrid(PARAM_GRID))} configs...")
        best_params = purged_cv_tune(X_train, y_train, n_splits=4)
        print(f"  Best params: {best_params}")

        # -- Train final model -----------------------------------------
        model = train_lgbm(X_train, y_train, best_params)

        # -- Feature importance for this fold --------------------------
        fi = pd.Series(model.feature_importances_, index=X_train.columns)
        feat_importances.append(fi)

        # -- Predict on OOS window -------------------------------------
        proba = model.predict_proba(X_test)[:, 1]  # P(up)
        lgbm_sig = pd.Series(0, index=test_dates)
        lgbm_sig[proba > PROBA_HIGH] = 1
        lgbm_sig[proba < PROBA_LOW] = -1

        # -- OOS accuracy (directional) --------------------------------
        pred_direction = (proba > 0.5).astype(int)
        accuracy = (pred_direction == y_test.values).mean()
        print(f"  OOS directional accuracy: {accuracy:.1%}")
        print(f"  LGBM signal breakdown: Long={( lgbm_sig==1).sum()}, "
              f"Short={(lgbm_sig==-1).sum()}, Flat={(lgbm_sig==0).sum()}")

        # -- Collect OOS signals ---------------------------------------
        all_oos_dates.extend(test_dates.tolist())
        all_lgbm_signals.extend(lgbm_sig.values.tolist())
        all_ma_signals.extend(ma_signal.reindex(test_dates).fillna(0).values.tolist())

        fold_num += 1
        test_start_idx += TEST_DAYS

    # -- Backtest both strategies on OOS-only period -------------------
    oos_idx = pd.DatetimeIndex(all_oos_dates)
    lgbm_sig_s = pd.Series(all_lgbm_signals, index=oos_idx)
    ma_sig_s = pd.Series(all_ma_signals, index=oos_idx)

    ohlcv_oos = ohlcv.reindex(oos_idx).dropna(subset=["close"])
    lgbm_sig_s = lgbm_sig_s.reindex(ohlcv_oos.index)
    ma_sig_s = ma_sig_s.reindex(ohlcv_oos.index)

    bt = Backtester(ohlcv_oos, COST_MODEL)
    lgbm_result = bt.run(lgbm_sig_s, "LightGBM")
    ma_result   = bt.run(ma_sig_s, "MA Crossover")
    bh_result   = buy_and_hold(ohlcv_oos, COST_MODEL)

    # -- Feature importance (averaged across folds) --------------------
    avg_fi = pd.concat(feat_importances, axis=1).mean(axis=1).sort_values(ascending=False)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY - {name} (OOS only)")
    print(f"{'='*60}")
    summary = bt.summary(lgbm_result, "LightGBM (net)")
    summary["MA Crossover (net)"] = bt.summary(ma_result, "MA Crossover (net)")["MA Crossover (net)"]
    summary["Buy & Hold"] = bt.summary(bh_result, "Buy & Hold")["Buy & Hold"]
    print(summary.to_string(index=False))

    print(f"\nTop 10 Features (avg importance across folds):")
    for feat, imp in avg_fi.head(10).items():
        print(f"  {feat:<35s} {imp:.1f}")

    return {
        "lgbm_result": lgbm_result,
        "ma_result": ma_result,
        "bh_result": bh_result,
        "avg_feature_importance": avg_fi,
        "summary": summary,
        "oos_index": oos_idx,
        "ohlcv_oos": ohlcv_oos,
    }


# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================

def save_results(results: dict, name: str):
    """Save all results to results/proposal1/<name>/"""
    out_dir = RESULTS_DIR / name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    lgbm_r = results["lgbm_result"]
    ma_r   = results["ma_result"]
    bh_r   = results["bh_result"]
    summary = results["summary"]
    avg_fi  = results["avg_feature_importance"]

    # Equity curves
    eq = pd.DataFrame({
        "lgbm_net": lgbm_r.equity_curve,
        "lgbm_gross": lgbm_r.equity_gross,
        "ma_net": ma_r.equity_curve,
        "ma_gross": ma_r.equity_gross,
        "buy_and_hold": bh_r.equity_curve,
    })
    eq.to_csv(out_dir / "equity_curves.csv")

    # Summary metrics
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    # Feature importance
    avg_fi.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(
        out_dir / "feature_importance.csv", index=False)

    # Trade logs
    def trades_to_df(result):
        return pd.DataFrame([{
            "entry_date": t.entry_date, "exit_date": t.exit_date,
            "direction": t.direction, "entry_price": t.entry_price,
            "exit_price": t.exit_price, "gross_pnl_pct": t.gross_pnl_pct,
            "net_pnl_pct": t.net_pnl_pct, "costs_pct": t.costs_pct
        } for t in result.trades])

    trades_to_df(lgbm_r).to_csv(out_dir / "trades_lgbm.csv", index=False)
    trades_to_df(ma_r).to_csv(out_dir / "trades_ma.csv", index=False)

    # Human-readable summary text
    with open(out_dir / "summary_report.txt", "w") as f:
        f.write(f"PROPOSAL 1 RESULTS - {name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("="*60 + "\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n\nTop 15 Features by Importance:\n")
        for feat, imp in avg_fi.head(15).items():
            f.write(f"  {feat:<40s} {imp:.1f}\n")
        f.write(f"\n\nCost drag (Gross->Net CAGR):\n")
        f.write(f"  LightGBM : {lgbm_r.cagr_gross:.1%} -> {lgbm_r.cagr_net:.1%} "
                f"(drag = {lgbm_r.cost_drag_cagr:.1%})\n")
        f.write(f"  MA Cross : {ma_r.cagr_gross:.1%} -> {ma_r.cagr_net:.1%} "
                f"(drag = {ma_r.cost_drag_cagr:.1%})\n")

    print(f"\n  All results saved to {out_dir}/")


# ==============================================================================
# MAIN
# ==============================================================================

def load_instrument_data(name: str):
    """Load OHLCV and features for an instrument."""
    ohlcv_f, fr_f, feat_f = INSTRUMENTS[name]

    ohlcv = pd.read_parquet(RAW_DIR / ohlcv_f)
    fr_path = RAW_DIR / fr_f
    if fr_path.exists():
        fr = pd.read_parquet(fr_path).squeeze()
        ohlcv["funding_rate"] = fr.reindex(ohlcv.index).fillna(0)
    else:
        ohlcv["funding_rate"] = 0.0

    feat_path = FEAT_DIR / feat_f
    if not feat_path.exists():
        print(f"  Features not found at {feat_path}. Computing now...")
        feats = build_features(ohlcv, name)
        feats.to_parquet(feat_path)
    else:
        feats = pd.read_parquet(feat_path)

    return ohlcv, feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proposal 1: LightGBM vs MA Crossover")
    parser.add_argument("--instruments", nargs="+", default=["BTC", "ETH"])
    args = parser.parse_args()

    all_results = {}
    for name in args.instruments:
        print(f"\n{'#'*60}")
        print(f"# INSTRUMENT: {name}")
        print(f"{'#'*60}")

        ohlcv, feats = load_instrument_data(name)
        results = run_walk_forward(ohlcv, feats, name)
        save_results(results, name)
        all_results[name] = results

    # -- Cross-instrument comparison -----------------------------------
    print(f"\n{'='*60}")
    print("CROSS-INSTRUMENT COMPARISON")
    print(f"{'='*60}")
    for name, res in all_results.items():
        lgbm = res["lgbm_result"]
        ma   = res["ma_result"]
        print(f"\n{name}:")
        print(f"  LightGBM: Sharpe={lgbm.sharpe_net:.2f} | CAGR={lgbm.cagr_net:.1%} | "
              f"MaxDD={lgbm.max_dd_net:.1%} | Trades={lgbm.n_trades}")
        print(f"  MA Cross: Sharpe={ma.sharpe_net:.2f}   | CAGR={ma.cagr_net:.1%} | "
              f"MaxDD={ma.max_dd_net:.1%} | Trades={ma.n_trades}")

    print(f"\nDone. All results in {RESULTS_DIR}/")
