"""
proposal1_main_v2.py
====================
Expanded Proposal 1: Walk-forward ML classification with LightGBM,
Random Forest, Logistic Regression, and Buy & Hold baselines on BTC/ETH.

Includes reproducible seeding (seed=42) and standardizes trading frictions
to 0.15% one-way cost (0.10% taker fee + 0.05% slippage).

Author: Ria Chawak | IIT Bombay Research Internship 2026
"""

import os
import random
import warnings
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

# Set random seeds for absolute reproducibility
random.seed(42)
np.random.seed(42)

# Local imports
from backtester import Backtester, CostModel, buy_and_hold
from features import build_features

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Standardized Frictions: 0.10% taker fee + 0.05% slippage = 0.15% one-way cost
COST_MODEL = CostModel(taker_fee=0.0010, slippage=0.0005)
RISK_FREE = 0.05

# Walk-Forward parameters
TRAIN_DAYS = 365
TEST_DAYS  = 90
EMBARGO    = 5
PROBA_HIGH = 0.55
PROBA_LOW  = 0.45

# LightGBM Params grid
LGB_GRID = {
    "learning_rate": [0.02, 0.05],
    "num_leaves": [20, 30],
}
LGB_FIXED = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
}

# Random Forest Params grid
RF_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [5, 8],
}

# Logistic Regression Params grid
LR_GRID = {
    "C": [0.01, 0.1, 1.0],
}

def make_binary_labels(close: pd.Series) -> pd.Series:
    fwd_ret = close.pct_change().shift(-1)
    labels = (fwd_ret > 0).astype(int)
    return labels

def tune_lgb(X_train, y_train, n_splits=4):
    n = len(X_train)
    fold_size = n // (n_splits + 1)
    best_score = np.inf
    best_params = list(ParameterGrid(LGB_GRID))[0]
    
    for params in ParameterGrid(LGB_GRID):
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
            
            all_params = {**LGB_FIXED, **params}
            model = lgb.LGBMClassifier(**all_params, n_estimators=100, random_state=42)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(15, verbose=False)])
            val_pred = model.predict_proba(X_val)[:, 1]
            score = log_loss(y_val, val_pred)
            fold_scores.append(score)
            
        avg_score = np.mean(fold_scores) if fold_scores else np.inf
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
            
    return best_params

def tune_rf(X_train, y_train, n_splits=4):
    n = len(X_train)
    fold_size = n // (n_splits + 1)
    best_score = np.inf
    best_params = list(ParameterGrid(RF_GRID))[0]
    
    for params in ParameterGrid(RF_GRID):
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
            
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr)
            val_pred = model.predict_proba(X_val)[:, 1]
            score = log_loss(y_val, val_pred)
            fold_scores.append(score)
            
        avg_score = np.mean(fold_scores) if fold_scores else np.inf
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
            
    return best_params

def tune_lr(X_train, y_train, n_splits=4):
    n = len(X_train)
    fold_size = n // (n_splits + 1)
    best_score = np.inf
    best_params = list(ParameterGrid(LR_GRID))[0]
    
    for params in ParameterGrid(LR_GRID):
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
            
            model = LogisticRegression(**params, random_state=42, max_iter=1000)
            model.fit(X_tr, y_tr)
            val_pred = model.predict_proba(X_val)[:, 1]
            score = log_loss(y_val, val_pred)
            fold_scores.append(score)
            
        avg_score = np.mean(fold_scores) if fold_scores else np.inf
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
            
    return best_params

def run_walk_forward_v2(ohlcv: pd.DataFrame, features: pd.DataFrame, name: str) -> dict:
    close = ohlcv["close"]
    labels = make_binary_labels(close)
    
    # Align
    features_clean = features.dropna(axis=1, how='all')
    valid_idx = features_clean.dropna().index.intersection(labels.dropna().index)
    X = features_clean.loc[valid_idx]
    y = labels.loc[valid_idx]
    
    test_start_idx = TRAIN_DAYS
    
    all_oos_dates = []
    
    # Prediction signals lists
    lgbm_sigs = []
    rf_sigs = []
    lr_sigs = []
    
    fold_num = 0
    
    print(f"\nWalk-Forward Pipeline v2: {name} | Train>={TRAIN_DAYS}d | Test={TEST_DAYS}d")
    
    while test_start_idx + TEST_DAYS <= len(X):
        train_end_idx = test_start_idx - EMBARGO
        train_end_idx = max(train_end_idx, TRAIN_DAYS)
        test_end_idx = min(test_start_idx + TEST_DAYS, len(X))
        
        X_train = X.iloc[:train_end_idx]
        y_train = y.iloc[:train_end_idx]
        X_test  = X.iloc[test_start_idx:test_end_idx]
        y_test  = y.iloc[test_start_idx:test_end_idx]
        test_dates = X.index[test_start_idx:test_end_idx]
        
        # 1. Tune & Train LightGBM
        best_lgb_params = tune_lgb(X_train, y_train)
        model_lgb = lgb.LGBMClassifier(**{**LGB_FIXED, **best_lgb_params}, n_estimators=100, random_state=42)
        model_lgb.fit(X_train, y_train)
        
        # 2. Tune & Train Random Forest
        best_rf_params = tune_rf(X_train, y_train)
        model_rf = RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1)
        model_rf.fit(X_train, y_train)
        
        # 3. Tune & Train Logistic Regression
        best_lr_params = tune_lr(X_train, y_train)
        model_lr = LogisticRegression(**best_lr_params, random_state=42, max_iter=1000)
        model_lr.fit(X_train, y_train)
        
        # Predict Probabilities
        p_lgb = model_lgb.predict_proba(X_test)[:, 1]
        p_rf  = model_rf.predict_proba(X_test)[:, 1]
        p_lr  = model_lr.predict_proba(X_test)[:, 1]
        
        # Turn to signals: trigger only on boundary crossing, else 0
        def proba_to_signal(prob):
            sig = pd.Series(0.0, index=test_dates)
            sig[prob > PROBA_HIGH] = 1.0
            sig[prob < PROBA_LOW] = -1.0
            return sig
            
        lgbm_sigs.extend(proba_to_signal(p_lgb).values.tolist())
        rf_sigs.extend(proba_to_signal(p_rf).values.tolist())
        lr_sigs.extend(proba_to_signal(p_lr).values.tolist())
        
        all_oos_dates.extend(test_dates.tolist())
        fold_num += 1
        test_start_idx += TEST_DAYS
        
    # Standardize backtesting on the OOS period
    oos_idx = pd.DatetimeIndex(all_oos_dates)
    ohlcv_oos = ohlcv.reindex(oos_idx).dropna(subset=["close"])
    
    # Save the signals to results for multi-engine verification later
    signals_df = pd.DataFrame({
        "Date": ohlcv_oos.index,
        "Close": ohlcv_oos["close"],
        "Signal_LGB": lgbm_sigs[:len(ohlcv_oos)],
        "Signal_RF": rf_sigs[:len(ohlcv_oos)],
        "Signal_LR": lr_sigs[:len(ohlcv_oos)],
        "Signal_BH": 1.0 # Always Long
    })
    
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    signals_df.to_csv(out_dir / f"signals_{name.lower()}_v2.csv", index=False)
    print(f"[+] Saved model signals to {out_dir / f'signals_{name.lower()}_v2.csv'}")
    
    # Execute Custom Backtester for all models
    bt = Backtester(ohlcv_oos, COST_MODEL)
    
    # Reconcile signals to trade only on signal crossings to avoid rebalancing fee-drag
    def run_without_rebalance(sig_series, label):
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
        return bt.run(trade_sig, label)
        
    res_lgb = run_without_rebalance(pd.Series(lgbm_sigs[:len(ohlcv_oos)], index=ohlcv_oos.index), "LightGBM")
    res_rf  = run_without_rebalance(pd.Series(rf_sigs[:len(ohlcv_oos)], index=ohlcv_oos.index), "Random Forest")
    res_lr  = run_without_rebalance(pd.Series(lr_sigs[:len(ohlcv_oos)], index=ohlcv_oos.index), "Logistic Regression")
    res_bh  = buy_and_hold(ohlcv_oos, COST_MODEL)
    
    summary = bt.summary(res_lgb, "LightGBM (net)")
    summary["Random Forest (net)"] = bt.summary(res_rf, "Random Forest (net)")["Random Forest (net)"]
    summary["Logistic Regression (net)"] = bt.summary(res_lr, "Logistic Regression (net)")["Logistic Regression (net)"]
    summary["Buy & Hold"] = bt.summary(res_bh, "Buy & Hold")["Buy & Hold"]
    
    print("\nCustom Backtester Net Results Comparison:")
    print(summary.to_string(index=False))
    
    return {
        "lgb_res": res_lgb,
        "rf_res": res_rf,
        "lr_res": res_lr,
        "bh_res": res_bh,
        "summary": summary,
        "ohlcv_oos": ohlcv_oos
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruments", nargs="+", default=["BTC", "ETH"])
    args = parser.parse_args()
    
    for name in args.instruments:
        print(f"\n{'-'*50}\nTraining Models for {name}\n{'-'*50}")
        # Load from daily CSV data directly
        csv_path = DATA_DIR / f"{name}USDT_daily.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing price file {csv_path}. Re-run copy scripts.")
            
        ohlcv = pd.read_csv(csv_path)
        # Ensure column names are lowercase for features.py
        ohlcv.columns = [c.lower() for c in ohlcv.columns]
        ohlcv["Date"] = pd.to_datetime(ohlcv["timestamp"])
        ohlcv = ohlcv.set_index("Date").sort_index()
        
        # Re-map OHLCV Close column name for features
        if "close" not in ohlcv.columns and "close_price" in ohlcv.columns:
            ohlcv = ohlcv.rename(columns={"close_price": "close"})
            
        # Build features on the fly
        feats = build_features(ohlcv, name)
        
        results = run_walk_forward_v2(ohlcv, feats, name)
        
        # Save custom curve results for IEEE papers
        out_sub = RESULTS_DIR / name.lower()
        out_sub.mkdir(parents=True, exist_ok=True)
        
        # Save equity curves
        curves_df = pd.DataFrame({
            "lgbm_net": results["lgb_res"].equity_curve,
            "rf_net": results["rf_res"].equity_curve,
            "lr_net": results["lr_res"].equity_curve,
            "buy_and_hold": results["bh_res"].equity_curve
        })
        curves_df.to_csv(out_sub / "equity_curves_v2.csv")
        results["summary"].to_csv(out_sub / "metrics_summary_v2.csv", index=False)
        print(f"[+] Out-of-sample v2 metrics saved to {out_sub / 'metrics_summary_v2.csv'}")
