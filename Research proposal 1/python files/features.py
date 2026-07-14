"""
features.py
===========
Computes all technical indicators and derivatives-specific features
for the ML pipeline. Each indicator has short / medium / long window variants.

Total features: 63+ (before any feature selection)

Important: ALL features are computed with only past data (no look-ahead).
Using pandas shift() and rolling() which are strictly backward-looking.

Usage:
    python features.py                      # compute for BTC and ETH
    python features.py --instruments BTC    # specific instrument

Author: Ria Chawak | IIT Bombay Research Internship 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

DATA_DIR = Path("data/raw")
FEAT_DIR = Path("data/features")
FEAT_DIR.mkdir(parents=True, exist_ok=True)

INSTRUMENTS = {
    "BTC": "BTC_USDT_ohlcv.parquet",
    "ETH": "ETH_USDT_ohlcv.parquet",
    "BNB": "BNB_USDT_ohlcv.parquet",
    "SOL": "SOL_USDT_ohlcv.parquet",
    "XRP": "XRP_USDT_ohlcv.parquet",
}
FUNDING_MAP = {k: v.replace("_ohlcv", "_funding") for k, v in INSTRUMENTS.items()}


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR FUNCTIONS
# Each returns a pd.Series or pd.DataFrame indexed same as input
# ══════════════════════════════════════════════════════════════════════════════

def ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False).mean()

def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def macd(close: pd.Series, fast: int, slow: int, signal: int):
    """Returns (macd_line, signal_line, histogram)"""
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    return macd_line, sig_line, hist

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    return 100 * (close - lowest_low) / denom

def roc(close: pd.Series, window: int) -> pd.Series:
    """Rate of Change = (close - close_n_bars_ago) / close_n_bars_ago"""
    return close.pct_change(window)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def realized_vol(log_ret: pd.Series, window: int) -> pd.Series:
    """Annualized realized volatility"""
    return log_ret.rolling(window).std() * np.sqrt(365)

def bollinger(close: pd.Series, window: int, n_std: float = 2.0):
    """Returns (upper, middle, lower, pct_b, width)"""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, mid, lower, pct_b, width

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()

def taker_buy_ratio(taker_buy_vol: pd.Series, total_vol: pd.Series, window: int) -> pd.Series:
    tbr = taker_buy_vol / total_vol.replace(0, np.nan)
    return tbr.rolling(window).mean()

def funding_rate_features(fr: pd.Series):
    """Compute multiple funding rate derived features."""
    feats = pd.DataFrame(index=fr.index)
    feats["fr_raw"] = fr
    feats["fr_5d_avg"] = fr.rolling(5).mean()
    feats["fr_7d_avg"] = fr.rolling(7).mean()
    feats["fr_20d_avg"] = fr.rolling(20).mean()
    # Z-scores
    for w in [7, 30, 90]:
        roll_mean = fr.rolling(w).mean()
        roll_std = fr.rolling(w).std().replace(0, np.nan)
        feats[f"fr_zscore_{w}d"] = (fr - roll_mean) / roll_std
    # Momentum of funding rate
    feats["fr_roc_1d"] = fr.diff(1)
    feats["fr_roc_3d"] = fr.diff(3)
    feats["fr_roc_7d"] = fr.diff(7)
    # Extreme funding flags
    feats["fr_high_flag"] = (feats["fr_zscore_30d"] > 2.0).astype(int)
    feats["fr_low_flag"] = (feats["fr_zscore_30d"] < -2.0).astype(int)
    return feats


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_features(ohlcv: pd.DataFrame, name: str = "ASSET") -> pd.DataFrame:
    """
    Build the full feature matrix for one instrument.
    Input: ohlcv DataFrame with columns [open, high, low, close, volume, funding_rate]
    Output: feature DataFrame, same index as input, all features strictly lagged.

    CRITICAL: all features at row t use only data from rows <= t.
    The prediction target (next-day direction) is added by labelling.py.
    """
    o, h, l, c, v = ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
    fr = ohlcv.get("funding_rate", pd.Series(0.0, index=ohlcv.index))
    tbv = ohlcv.get("taker_buy_volume", pd.Series(np.nan, index=ohlcv.index))

    log_ret = np.log(c / c.shift(1))
    feats = pd.DataFrame(index=ohlcv.index)

    # ── 1. Returns (lagged, to avoid look-ahead) ───────────────────────
    for lag in [1, 2, 3, 5, 7, 10, 14, 21]:
        feats[f"log_ret_lag{lag}"] = log_ret.shift(lag)
    # Simple pct change lagged
    for lag in [1, 5, 10, 20]:
        feats[f"pct_ret_lag{lag}"] = c.pct_change().shift(lag)

    # ── 2. EMA trend features ──────────────────────────────────────────
    for w in [9, 21, 50]:
        e = ema(c, w)
        feats[f"ema{w}"] = e
        feats[f"price_vs_ema{w}"] = (c / e) - 1   # how extended price is from EMA

    for w in [20, 50, 200]:
        feats[f"sma{w}"] = sma(c, w)
        feats[f"price_vs_sma{w}"] = (c / sma(c, w)) - 1

    # EMA crossover signals (sign of difference)
    feats["ema9_vs_ema21"] = np.sign(ema(c, 9) - ema(c, 21))
    feats["ema21_vs_ema50"] = np.sign(ema(c, 21) - ema(c, 50))
    feats["ema50_vs_sma200"] = np.sign(ema(c, 50) - sma(c, 200))  # core MA baseline signal

    # ── 3. MACD (3 variations) ─────────────────────────────────────────
    for (fast, slow, sig) in [(5, 13, 4), (12, 26, 9), (26, 50, 15)]:
        ml, sl, hist = macd(c, fast, slow, sig)
        feats[f"macd_line_{fast}_{slow}"] = ml
        feats[f"macd_hist_{fast}_{slow}"] = hist
        feats[f"macd_cross_{fast}_{slow}"] = np.sign(ml - sl)

    # ── 4. RSI (3 windows) ─────────────────────────────────────────────
    for w in [7, 14, 28]:
        feats[f"rsi_{w}"] = rsi(c, w)
        feats[f"rsi_{w}_vs50"] = feats[f"rsi_{w}"] - 50  # deviation from neutral

    # ── 5. Stochastic (3 windows) ─────────────────────────────────────
    for w in [5, 14, 21]:
        feats[f"stoch_{w}"] = stochastic(h, l, c, w)

    # ── 6. Rate of Change / Momentum ──────────────────────────────────
    for w in [5, 10, 20, 60]:
        feats[f"roc_{w}"] = roc(c, w)

    # ── 7. Volatility features ─────────────────────────────────────────
    for w in [5, 10, 20, 60]:
        feats[f"rv_{w}"] = realized_vol(log_ret, w)

    feats["rv_ratio_5_20"] = feats["rv_5"] / feats["rv_20"].replace(0, np.nan)
    feats["rv_ratio_10_20"] = feats["rv_10"] / feats["rv_20"].replace(0, np.nan)
    feats["rv_ratio_5_60"] = feats["rv_5"] / feats["rv_60"].replace(0, np.nan)
    feats["high_vol_regime"] = (feats["rv_20"] > feats["rv_20"].rolling(252).quantile(0.75)).astype(int)

    # ATR (3 windows)
    for w in [7, 14, 21]:
        feats[f"atr_{w}"] = atr(h, l, c, w)
        feats[f"atr_{w}_pct"] = feats[f"atr_{w}"] / c  # normalise by price

    # ── 8. Bollinger Bands ─────────────────────────────────────────────
    for w, ns in [(10, 1.5), (20, 2.0), (50, 2.0)]:
        up, mid, lo, pctb, width = bollinger(c, w, ns)
        feats[f"bb_pctb_{w}"] = pctb
        feats[f"bb_width_{w}"] = width

    # ── 9. Volume features ─────────────────────────────────────────────
    vol_log = np.log(v + 1)
    for w in [5, 20, 60]:
        roll_mean = vol_log.rolling(w).mean()
        roll_std = vol_log.rolling(w).std().replace(0, np.nan)
        feats[f"vol_zscore_{w}"] = (vol_log - roll_mean) / roll_std

    obv_series = obv(c, v)
    for w in [5, 10, 20]:
        feats[f"obv_roc_{w}"] = obv_series.pct_change(w)

    # Taker buy ratio (if available, else fill with NaN — model handles it)
    for w in [5, 10, 20]:
        feats[f"tbr_{w}"] = taker_buy_ratio(tbv, v, w)

    # ── 10. Funding rate features (unique to perpetuals) ───────────────
    fr_feats = funding_rate_features(fr)
    for col in fr_feats.columns:
        feats[col] = fr_feats[col]

    # ── 11. Calendar features ──────────────────────────────────────────
    feats["day_of_week"] = feats.index.dayofweek
    feats["month"] = feats.index.month
    feats["day_of_week_sin"] = np.sin(2 * np.pi * feats["day_of_week"] / 7)
    feats["day_of_week_cos"] = np.cos(2 * np.pi * feats["day_of_week"] / 7)
    feats["month_sin"] = np.sin(2 * np.pi * feats["month"] / 12)
    feats["month_cos"] = np.cos(2 * np.pi * feats["month"] / 12)

    # ── 12. Range / candlestick features ──────────────────────────────
    feats["daily_range_pct"] = (h - l) / c
    feats["upper_shadow_pct"] = (h - c.clip(upper=o)) / c  # approx
    feats["lower_shadow_pct"] = (c.clip(upper=o) - l) / c
    feats["body_pct"] = (c - o).abs() / c

    # ── Final: shift all features by 1 to avoid look-ahead ────────────
    # Feature at row t must be computed from data available at close of bar t.
    # Since we compute everything from historical closes, EMAs etc., the features
    # at row t already use only data up to and including t.
    # However, lagged returns already include shift() in their definition.
    # We do NOT shift the feature matrix here — the labelling step
    # in labelling.py uses label[t] = sign(close[t+1] - close[t]),
    # so the model trains: features[t] → label[t] (what happens next day).
    # Execution happens at open of bar t+1.

    # Drop the raw OHLCV columns (not features themselves)
    feats = feats.replace([np.inf, -np.inf], np.nan)

    n_feat = feats.shape[1]
    n_nan = feats.isnull().sum().sum()
    print(f"  {name}: {n_feat} features, {n_nan} NaN values "
          f"(expected from rolling warm-up period)")
    return feats


def run_all(instruments=None):
    instruments = instruments or list(INSTRUMENTS.keys())
    for name in instruments:
        ohlcv_path = DATA_DIR / INSTRUMENTS[name]
        if not ohlcv_path.exists():
            print(f"  WARNING: {ohlcv_path} not found. Run data_loader.py first.")
            continue
        print(f"\nBuilding features for {name}...")
        ohlcv = pd.read_parquet(ohlcv_path)

        # Load funding rate
        fr_path = DATA_DIR / FUNDING_MAP[name]
        if fr_path.exists():
            fr = pd.read_parquet(fr_path).squeeze()
            ohlcv["funding_rate"] = fr.reindex(ohlcv.index).fillna(0)
        else:
            ohlcv["funding_rate"] = 0.0

        feats = build_features(ohlcv, name)

        out_path = FEAT_DIR / f"{name}_features.parquet"
        feats.to_parquet(out_path)
        print(f"  Saved to {out_path}")
        print(f"  Feature matrix shape: {feats.shape}")
        print(f"  First 5 features: {list(feats.columns[:5])}")
        print(f"  Last 5 features:  {list(feats.columns[-5:])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruments", nargs="+", default=None)
    args = parser.parse_args()
    run_all(args.instruments)
