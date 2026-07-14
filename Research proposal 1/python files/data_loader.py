"""
data_loader.py
==============
Downloads daily OHLCV + funding rate data from Binance Futures via ccxt.
Saves to data/raw/ as parquet files. Has a refresh mode to pull only new bars.

Usage:
    python data_loader.py                          # download all default instruments
    python data_loader.py --instruments BTC ETH    # specific instruments
    python data_loader.py --start 2020-01-01       # from a specific date
    python data_loader.py --refresh                # only fetch new bars since last save

Author: Ria Chawak | IIT Bombay Research Internship 2026
"""

import ccxt
import pandas as pd
import numpy as np
import os
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INSTRUMENTS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT",
                       "SOL/USDT:USDT", "XRP/USDT:USDT"]
DEFAULT_START = "2020-01-01"
TIMEFRAME = "1d"
LIMIT = 1000  # max bars per API call
SLEEP_SECS = 0.3  # rate limit


def get_exchange():
    """Initialise Binance futures exchange (unauthenticated for public data)."""
    exchange = ccxt.binance({
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
    })
    exchange.load_markets()
    return exchange


def fetch_ohlcv(exchange, symbol: str, start: str, end: str = None) -> pd.DataFrame:
    """
    Fetch all daily OHLCV bars for a symbol from start to end (or now).
    Returns DataFrame with columns: timestamp, open, high, low, close, volume.

    No look-ahead: each bar's timestamp is the START of that day (00:00 UTC).
    The close price is known only at 23:59 UTC that day.
    """
    since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else None

    all_bars = []
    while True:
        bars = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, since=since_ms, limit=LIMIT)
        if not bars:
            break
        all_bars.extend(bars)
        last_ts = bars[-1][0]
        if end_ms and last_ts >= end_ms:
            break
        if len(bars) < LIMIT:
            break
        since_ms = last_ts + 1
        time.sleep(SLEEP_SECS)

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.normalize()
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Sanity check: no NaNs, no close=0
    assert df.isnull().sum().sum() == 0, f"NaNs found in OHLCV for {symbol}"
    assert (df["close"] > 0).all(), f"Zero/negative close price found for {symbol}"

    print(f"  {symbol}: {len(df)} bars [{df.index[0].date()} -> {df.index[-1].date()}]")
    return df


def fetch_funding_rate(exchange, symbol: str, start: str, end: str = None) -> pd.Series:
    """
    Fetch 8-hourly funding rate for a perpetual symbol and aggregate to daily.
    Daily funding = sum of 3 × 8h rates per day (funding paid 3 times/day on Binance).

    Returns pd.Series indexed by date (UTC-normalised daily).

    Note: Binance provides historical funding rates via /fapi/v1/fundingRate.
    ccxt wraps this as fetch_funding_rate_history.
    """
    since_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else None

    all_rates = []
    while True:
        try:
            rates = exchange.fetch_funding_rate_history(symbol, since=since_ms, limit=1000)
        except Exception as e:
            print(f"  Warning: could not fetch funding rate for {symbol}: {e}")
            return pd.Series(dtype=float)

        if not rates:
            break
        all_rates.extend(rates)
        last_ts = rates[-1]["timestamp"]
        if end_ms and last_ts >= end_ms:
            break
        if len(rates) < 1000:
            break
        since_ms = last_ts + 1
        time.sleep(SLEEP_SECS)

    if not all_rates:
        return pd.Series(dtype=float)

    fr_df = pd.DataFrame(all_rates)[["timestamp", "fundingRate"]]
    fr_df["timestamp"] = pd.to_datetime(fr_df["timestamp"], unit="ms", utc=True)
    fr_df["date"] = fr_df["timestamp"].dt.normalize()
    # Sum the 3 intra-day rates to get daily funding cost/income
    daily_fr = fr_df.groupby("date")["fundingRate"].sum()
    daily_fr.index.name = "timestamp"
    print(f"  {symbol} funding rate: {len(daily_fr)} daily bars")
    return daily_fr


def load_or_download(symbol: str, start: str, refresh: bool = False) -> pd.DataFrame:
    """
    Load from parquet if exists and not refreshing, else download and save.
    In refresh mode, fetches only new bars after the last saved date.
    """
    # Convert symbol to a safe filename: BTC/USDT:USDT -> BTC_USDT
    fname = symbol.split("/")[0] + "_" + symbol.split("/")[1].split(":")[0]
    ohlcv_path = DATA_DIR / f"{fname}_ohlcv.parquet"
    fr_path = DATA_DIR / f"{fname}_funding.parquet"

    exchange = get_exchange()

    # ── OHLCV ──────────────────────────────────────────────────────────
    if ohlcv_path.exists() and not refresh:
        ohlcv = pd.read_parquet(ohlcv_path)
        print(f"  Loaded {fname} OHLCV from cache ({len(ohlcv)} bars)")
    else:
        fetch_start = start
        if ohlcv_path.exists() and refresh:
            existing = pd.read_parquet(ohlcv_path)
            fetch_start = str(existing.index[-1].date())
            print(f"  Refreshing {fname} OHLCV from {fetch_start}...")
        else:
            existing = None
            print(f"  Downloading {fname} OHLCV from {fetch_start}...")

        new_bars = fetch_ohlcv(exchange, symbol, fetch_start)
        if existing is not None:
            ohlcv = pd.concat([existing, new_bars]).loc[~pd.concat([existing, new_bars]).index.duplicated(keep="last")]
        else:
            ohlcv = new_bars
        ohlcv.to_parquet(ohlcv_path)

    # ── Funding Rate ───────────────────────────────────────────────────
    if fr_path.exists() and not refresh:
        fr = pd.read_parquet(fr_path).squeeze()
        print(f"  Loaded {fname} funding rate from cache ({len(fr)} bars)")
    else:
        fetch_start = start
        if fr_path.exists() and refresh:
            existing_fr = pd.read_parquet(fr_path).squeeze()
            fetch_start = str(existing_fr.index[-1].date())
        else:
            existing_fr = None

        new_fr = fetch_funding_rate(exchange, symbol, fetch_start)
        if existing_fr is not None and len(new_fr) > 0:
            fr = pd.concat([existing_fr, new_fr]).loc[~pd.concat([existing_fr, new_fr]).index.duplicated(keep="last")]
        elif len(new_fr) > 0:
            fr = new_fr
        else:
            fr = existing_fr if existing_fr is not None else pd.Series(dtype=float)

        if len(fr) > 0:
            fr.to_frame("funding_rate").to_parquet(fr_path)

    # ── Merge ──────────────────────────────────────────────────────────
    df = ohlcv.copy()
    if len(fr) > 0:
        df["funding_rate"] = fr.reindex(df.index).fillna(method="ffill").fillna(0)
    else:
        df["funding_rate"] = 0.0

    # Additional Binance-specific columns from klines (taker buy volume)
    # These are available in the raw API but ccxt strips them by default
    # We add placeholders here; for real runs, use requests library to call
    # /fapi/v1/klines directly and capture columns 9 (taker buy base vol)
    if "taker_buy_volume" not in df.columns:
        df["taker_buy_volume"] = np.nan  # populate with direct API call in production

    return df


def run_data_validation(df: pd.DataFrame, symbol: str):
    """
    Data quality checks — run after downloading.
    Checks: no missing values, no duplicate dates, no zero/negative prices,
    high < low check, date gaps.
    """
    print(f"\n  Validating {symbol}...")
    assert not df.index.duplicated().any(), "Duplicate timestamps found!"
    assert df[["open","high","low","close"]].isnull().sum().sum() == 0, "NaNs in OHLC!"
    assert (df["close"] > 0).all(), "Zero/negative close!"
    assert (df["high"] >= df["low"]).all(), "High < Low found!"

    # Check for date gaps (crypto runs 24/7, so every calendar day should exist)
    date_range = pd.date_range(df.index[0], df.index[-1], freq="D", tz="UTC")
    missing = date_range.difference(df.index)
    if len(missing) > 0:
        print(f"  WARNING: {len(missing)} missing dates: {missing[:5]}...")
    else:
        print(f"  Date continuity: OK (no gaps)")

    # Volume check
    zero_vol = (df["volume"] == 0).sum()
    if zero_vol > 0:
        print(f"  WARNING: {zero_vol} bars with zero volume")

    print(f"  All basic checks passed. Shape: {df.shape}")
    print(f"  Date range: {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"  Close price range: {df['close'].min():.2f} – {df['close'].max():.2f}")
    print(f"  Funding rate range: {df['funding_rate'].min():.6f} – {df['funding_rate'].max():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance futures data")
    parser.add_argument("--instruments", nargs="+", default=None)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    instruments = args.instruments or DEFAULT_INSTRUMENTS
    # Convert short names to ccxt symbols
    sym_map = {"BTC": "BTC/USDT:USDT", "ETH": "ETH/USDT:USDT",
               "BNB": "BNB/USDT:USDT", "SOL": "SOL/USDT:USDT",
               "XRP": "XRP/USDT:USDT"}
    symbols = [sym_map.get(i, i) for i in instruments]

    print(f"Downloading {len(symbols)} instruments from {args.start}...\n")
    all_data = {}
    for sym in symbols:
        print(f"-> {sym}")
        df = load_or_download(sym, args.start, refresh=args.refresh)
        run_data_validation(df, sym)
        all_data[sym] = df
        print()

    print(f"\nAll data saved to {DATA_DIR}/")
    print("Files created:")
    for f in sorted(DATA_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:40s}  {size_kb:.1f} KB")
