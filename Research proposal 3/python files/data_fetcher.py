import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_binance_daily_prices(symbol, start_date_str, end_date_str):
    """
    Fetches daily (1d) price data from Binance public Spot API.
    """
    print(f"[*] Extracting historical price stream for {symbol}...")
    url = "https://api.binance.com/api/v3/klines"
    
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    start_time = int(start_dt.timestamp() * 1000)
    end_time = int(end_dt.timestamp() * 1000)
    
    all_data = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[!] Error fetching price data for {symbol}: {e}")
            break
            
        if not data:
            break
            
        all_data.extend(data)
        current_start = data[-1][6] + 1
        time.sleep(0.1)
        
    if not all_data:
        print(f"[!] No price data retrieved for {symbol}.")
        return None
        
    df = pd.DataFrame(all_data, columns=[
        "Open_Time", "Open", "High", "Low", "Close", "Volume",
        "Close_Time", "Quote_Asset_Volume", "Number_of_Trades",
        "Taker_Buy_Base", "Taker_Buy_Quote", "Ignore"
    ])
    
    df["Date"] = pd.to_datetime(df["Open_Time"], unit="ms").dt.date
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
        
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    return df

def fetch_binance_funding_rates(symbol, start_date_str, end_date_str):
    """
    Fetches 8-hour historical funding rate data from Binance Futures API.
    """
    print(f"[*] Extracting historical funding rate stream for {symbol}...")
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    start_time = int(start_dt.timestamp() * 1000)
    end_time = int(end_dt.timestamp() * 1000)
    
    all_data = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[!] Error fetching funding rates for {symbol}: {e}")
            break
            
        if not data:
            break
            
        all_data.extend(data)
        current_start = data[-1]["fundingTime"] + 1
        time.sleep(0.1)
        
    if not all_data:
        print(f"[!] No funding rate data retrieved for {symbol}.")
        return None
        
    df = pd.DataFrame(all_data)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["Date"] = pd.to_datetime(df["fundingTime"], unit="ms").dt.date
    df = df.sort_values("fundingTime").reset_index(drop=True)
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def main():
    start_date = "2020-01-01"
    end_date = "2026-06-01" # Fetch through end of May 2026
    
    # 1. Fetch Spot Price Data
    price_df = fetch_binance_daily_prices("BTCUSDT", start_date, end_date)
    if price_df is None:
        print("[!] Failed to fetch price data. Aborting.")
        return
        
    # 2. Fetch Futures Funding Rate Data
    funding_df = fetch_binance_funding_rates("BTCUSDT", start_date, end_date)
    if funding_df is None:
        print("[!] Failed to fetch funding rate data. Aborting.")
        return
        
    # 3. Aggregate funding rate to daily (sum of all funding rates paid on that day)
    print("[*] Aggregating funding rate to daily frequency...")
    daily_funding = funding_df.groupby("Date")["fundingRate"].sum().reset_index()
    daily_funding.rename(columns={"fundingRate": "funding_rate"}, inplace=True)
    
    # 4. Merge price and daily funding rate on Date
    print("[*] Merging price and daily funding rate datasets...")
    merged_df = pd.merge(price_df, daily_funding, on="Date", how="inner")
    merged_df["Date"] = pd.to_datetime(merged_df["Date"])
    merged_df = merged_df.sort_values("Date").reset_index(drop=True)
    
    # 5. Feature Engineering with 3 window variations
    print("[*] Computing indicators with short, medium, and long window variations...")
    
    # Funding Rate rolling mean (3d, 14d, 30d)
    merged_df["funding_mean_3"] = merged_df["funding_rate"].rolling(3).mean()
    merged_df["funding_mean_14"] = merged_df["funding_rate"].rolling(14).mean()
    merged_df["funding_mean_30"] = merged_df["funding_rate"].rolling(30).mean()
    
    # Funding Rate rolling volatility (3d, 14d, 30d)
    merged_df["funding_vol_3"] = merged_df["funding_rate"].rolling(3).std()
    merged_df["funding_vol_14"] = merged_df["funding_rate"].rolling(14).std()
    merged_df["funding_vol_30"] = merged_df["funding_rate"].rolling(30).std()
    
    # Funding Rate rolling Z-score (5d, 20d, 60d)
    merged_df["funding_z_5"] = (merged_df["funding_rate"] - merged_df["funding_rate"].rolling(5).mean()) / (merged_df["funding_rate"].rolling(5).std() + 1e-8)
    merged_df["funding_z_20"] = (merged_df["funding_rate"] - merged_df["funding_rate"].rolling(20).mean()) / (merged_df["funding_rate"].rolling(20).std() + 1e-8)
    merged_df["funding_z_60"] = (merged_df["funding_rate"] - merged_df["funding_rate"].rolling(60).mean()) / (merged_df["funding_rate"].rolling(60).std() + 1e-8)
    
    # BTC Price Volatility - std of log returns (5d, 20d, 60d)
    btc_ret = np.log(merged_df["Close"] / merged_df["Close"].shift(1))
    merged_df["btc_vol_5"] = btc_ret.rolling(5).std() * np.sqrt(365)
    merged_df["btc_vol_20"] = btc_ret.rolling(20).std() * np.sqrt(365)
    merged_df["btc_vol_60"] = btc_ret.rolling(60).std() * np.sqrt(365)
    
    # BTC Price RSI (7d, 14d, 28d)
    merged_df["btc_rsi_7"] = compute_rsi(merged_df["Close"], 7)
    merged_df["btc_rsi_14"] = compute_rsi(merged_df["Close"], 14)
    merged_df["btc_rsi_28"] = compute_rsi(merged_df["Close"], 28)
    
    # BTC Price SMA (10d, 50d, 200d)
    merged_df["btc_sma_10"] = merged_df["Close"].rolling(10).mean()
    merged_df["btc_sma_50"] = merged_df["Close"].rolling(50).mean()
    merged_df["btc_sma_200"] = merged_df["Close"].rolling(200).mean()
    
    # Save cache
    output_path = os.path.join(DATA_DIR, "btc_funding_daily.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"[+] Combined data and indicators cached successfully at {output_path}")
    print(f"    - Date Range: {merged_df['Date'].min().strftime('%Y-%m-%d')} to {merged_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"    - Shape: {merged_df.shape}")

if __name__ == "__main__":
    main()
