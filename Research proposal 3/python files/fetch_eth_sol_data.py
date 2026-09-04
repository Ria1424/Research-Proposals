"""
fetch_eth_sol_data.py
=====================
Fetches real Binance perpetual futures data for ETH and SOL:
  - Daily OHLCV price (from Binance Spot API)
  - 8-hour funding rate history (from Binance Futures API)
  - Aggregates to daily funding rate and saves to:
      data/eth_funding_daily.csv
      data/sol_funding_daily.csv

Note: SOL perpetual on Binance launched 2020-09-14. Data before
this date is not available and will be filled from launch onwards.

Author: Ria Chawak | IIT Bombay Research Internship 2026
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2020-01-01"
END_DATE = "2026-06-01"


def fetch_binance_daily_prices(symbol: str, start_date_str: str, end_date_str: str) -> pd.DataFrame | None:
    """Fetch daily OHLCV from Binance Spot API (paginated)."""
    print(f"[*] Fetching price data: {symbol} ({start_date_str} to {end_date_str})")
    url = "https://api.binance.com/api/v3/klines"

    start_ms = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp() * 1000)
    end_ms   = int(datetime.strptime(end_date_str,   "%Y-%m-%d").timestamp() * 1000)

    all_rows = []
    cur = start_ms
    while cur < end_ms:
        try:
            r = requests.get(url, params={
                "symbol": symbol, "interval": "1d",
                "startTime": cur, "endTime": end_ms, "limit": 1000
            }, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  [!] Error: {e}")
            break
        if not data:
            break
        all_rows.extend(data)
        cur = data[-1][6] + 1       # next ms after close_time
        time.sleep(0.15)

    if not all_rows:
        print(f"  [!] No price data for {symbol}")
        return None

    df = pd.DataFrame(all_rows, columns=[
        "Open_Time", "Open", "High", "Low", "Close", "Volume",
        "Close_Time", "Quote_Asset_Volume", "Number_of_Trades",
        "Taker_Buy_Base", "Taker_Buy_Quote", "Ignore"
    ])
    df["Date"] = pd.to_datetime(df["Open_Time"], unit="ms").dt.date
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(float)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    print(f"  [+] {len(df)} daily price bars downloaded for {symbol}")
    return df


def fetch_binance_funding_rates(symbol: str, start_date_str: str, end_date_str: str) -> pd.DataFrame | None:
    """Fetch 8-hour funding rate history from Binance Futures API (paginated)."""
    print(f"[*] Fetching funding rates: {symbol} ({start_date_str} to {end_date_str})")
    url = "https://fapi.binance.com/fapi/v1/fundingRate"

    start_ms = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp() * 1000)
    end_ms   = int(datetime.strptime(end_date_str,   "%Y-%m-%d").timestamp() * 1000)

    all_rows = []
    cur = start_ms
    while cur < end_ms:
        try:
            r = requests.get(url, params={
                "symbol": symbol,
                "startTime": cur, "endTime": end_ms, "limit": 1000
            }, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  [!] Error: {e}")
            break
        if not data:
            break
        all_rows.extend(data)
        cur = data[-1]["fundingTime"] + 1
        time.sleep(0.15)

    if not all_rows:
        print(f"  [!] No funding rate data for {symbol}")
        return None

    df = pd.DataFrame(all_rows)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["Date"] = pd.to_datetime(df["fundingTime"], unit="ms").dt.date
    df = df.sort_values("fundingTime").reset_index(drop=True)
    print(f"  [+] {len(df)} 8-hour funding rate records downloaded for {symbol}")
    return df


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def build_dataset(spot_symbol: str, perp_symbol: str, out_name: str) -> bool:
    """Build complete dataset for one asset and save to data/."""
    price_df = fetch_binance_daily_prices(spot_symbol, START_DATE, END_DATE)
    if price_df is None:
        return False

    fr_df = fetch_binance_funding_rates(perp_symbol, START_DATE, END_DATE)
    if fr_df is None:
        return False

    # Aggregate funding rates to daily (sum of up to 3 payments per day)
    daily_fr = fr_df.groupby("Date")["fundingRate"].sum().reset_index()
    daily_fr.rename(columns={"fundingRate": "funding_rate"}, inplace=True)

    merged = pd.merge(price_df, daily_fr, on="Date", how="inner")
    merged["Date"] = pd.to_datetime(merged["Date"])
    merged = merged.sort_values("Date").reset_index(drop=True)

    # --- Feature Engineering (mirrors btc_funding_daily.csv schema) ---
    merged["funding_mean_3"]  = merged["funding_rate"].rolling(3).mean()
    merged["funding_mean_14"] = merged["funding_rate"].rolling(14).mean()
    merged["funding_mean_30"] = merged["funding_rate"].rolling(30).mean()

    merged["funding_vol_3"]   = merged["funding_rate"].rolling(3).std()
    merged["funding_vol_14"]  = merged["funding_rate"].rolling(14).std()
    merged["funding_vol_30"]  = merged["funding_rate"].rolling(30).std()

    merged["funding_z_5"]  = (merged["funding_rate"] - merged["funding_rate"].rolling(5).mean()) / \
                              (merged["funding_rate"].rolling(5).std() + 1e-8)
    merged["funding_z_20"] = (merged["funding_rate"] - merged["funding_rate"].rolling(20).mean()) / \
                              (merged["funding_rate"].rolling(20).std() + 1e-8)
    merged["funding_z_60"] = (merged["funding_rate"] - merged["funding_rate"].rolling(60).mean()) / \
                              (merged["funding_rate"].rolling(60).std() + 1e-8)

    log_ret = np.log(merged["Close"] / merged["Close"].shift(1))
    merged["price_vol_5"]  = log_ret.rolling(5).std()  * np.sqrt(365)
    merged["price_vol_20"] = log_ret.rolling(20).std() * np.sqrt(365)
    merged["price_vol_60"] = log_ret.rolling(60).std() * np.sqrt(365)

    merged["rsi_7"]  = compute_rsi(merged["Close"], 7)
    merged["rsi_14"] = compute_rsi(merged["Close"], 14)
    merged["rsi_28"] = compute_rsi(merged["Close"], 28)

    merged["sma_10"]  = merged["Close"].rolling(10).mean()
    merged["sma_50"]  = merged["Close"].rolling(50).mean()
    merged["sma_200"] = merged["Close"].rolling(200).mean()

    out_path = DATA_DIR / out_name
    merged.to_csv(out_path, index=False)
    print(f"[+] Saved {out_name}: {len(merged)} rows, {merged['Date'].min().date()} to {merged['Date'].max().date()}")
    return True


def main():
    print("=" * 60)
    print("  FETCHING REAL ETH & SOL PERPETUAL FUNDING RATE DATA")
    print("=" * 60)

    ok_eth = build_dataset("ETHUSDT", "ETHUSDT", "eth_funding_daily.csv")
    print()
    ok_sol = build_dataset("SOLUSDT", "SOLUSDT", "sol_funding_daily.csv")
    print()

    if ok_eth and ok_sol:
        print("[+] Both ETH and SOL datasets downloaded and saved successfully.")
    else:
        print("[!] One or more datasets failed to download. Check internet connection.")


if __name__ == "__main__":
    main()
