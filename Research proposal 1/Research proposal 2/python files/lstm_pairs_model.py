import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Directory Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_data():
    btc_path = os.path.join(DATA_DIR, "BTCUSDT_daily.csv")
    eth_path = os.path.join(DATA_DIR, "ETHUSDT_daily.csv")
    
    btc = pd.read_csv(btc_path)
    eth = pd.read_csv(eth_path)
    
    # Merge on Date
    df = pd.merge(btc[["Date", "Open", "Close", "Volume"]], eth[["Date", "Open", "Close", "Volume"]], on="Date", suffixes=("_btc", "_eth"))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def prepare_features(df):
    print("[*] Computing cointegrated spread (rolling 60-day OLS)...")
    y = np.log(df["Close_eth"].values)
    x = np.log(df["Close_btc"].values)
    
    y_series = pd.Series(y)
    x_series = pd.Series(x)
    
    # Rolling OLS formula: y = beta * x + alpha
    window = 60
    mean_y = y_series.rolling(window).mean()
    mean_x = x_series.rolling(window).mean()
    mean_xy = (y_series * x_series).rolling(window).mean()
    mean_x2 = (x_series ** 2).rolling(window).mean()
    
    cov_xy = mean_xy - mean_x * mean_y
    var_x = mean_x2 - mean_x ** 2
    
    # Handle division by zero
    var_x = var_x.replace(0, np.nan)
    beta = cov_xy / var_x
    beta = beta.ffill().bfill()
    alpha = mean_y - beta * mean_x
    alpha = alpha.ffill().bfill()
    
    spread = y - beta * x - alpha
    spread_series = pd.Series(spread)
    
    mean_spread = spread_series.rolling(window).mean()
    std_spread = spread_series.rolling(window).std()
    z_score = (spread_series - mean_spread) / (std_spread + 1e-8)
    
    df["beta"] = beta
    df["alpha"] = alpha
    df["spread"] = spread
    df["z_score"] = z_score
    
    print("[*] Engineering features with 3 window variations (short, medium, long)...")
    # 1. Spread Rolling Mean (5, 15, 30 days)
    df["spread_mean_5"] = spread_series.rolling(5).mean()
    df["spread_mean_15"] = spread_series.rolling(15).mean()
    df["spread_mean_30"] = spread_series.rolling(30).mean()
    
    # 2. Spread Rolling Volatility (5, 15, 30 days)
    df["spread_vol_5"] = spread_series.rolling(5).std()
    df["spread_vol_15"] = spread_series.rolling(15).std()
    df["spread_vol_30"] = spread_series.rolling(30).std()
    
    # 3. Spread RSI (7, 14, 28 days)
    df["spread_rsi_7"] = compute_rsi(spread_series, 7)
    df["spread_rsi_14"] = compute_rsi(spread_series, 14)
    df["spread_rsi_28"] = compute_rsi(spread_series, 28)
    
    # 4. Spread Momentum (ROC) - simple difference for spread (3, 10, 30 days)
    df["spread_roc_3"] = spread_series - spread_series.shift(3)
    df["spread_roc_10"] = spread_series - spread_series.shift(10)
    df["spread_roc_30"] = spread_series - spread_series.shift(30)
    
    # 5. Spread Rolling Z-score (10, 30, 60 days)
    df["spread_z_10"] = (spread_series - spread_series.rolling(10).mean()) / (spread_series.rolling(10).std() + 1e-8)
    df["spread_z_30"] = (spread_series - spread_series.rolling(30).mean()) / (spread_series.rolling(30).std() + 1e-8)
    df["spread_z_60"] = z_score
    
    # 6. BTC Log returns volatility (5, 15, 30 days)
    btc_ret = np.log(df["Close_btc"] / df["Close_btc"].shift(1))
    df["btc_vol_5"] = btc_ret.rolling(5).std() * np.sqrt(365.25)
    df["btc_vol_15"] = btc_ret.rolling(15).std() * np.sqrt(365.25)
    df["btc_vol_30"] = btc_ret.rolling(30).std() * np.sqrt(365.25)
    
    # 7. ETH Log returns volatility (5, 15, 30 days)
    eth_ret = np.log(df["Close_eth"] / df["Close_eth"].shift(1))
    df["eth_vol_5"] = eth_ret.rolling(5).std() * np.sqrt(365.25)
    df["eth_vol_15"] = eth_ret.rolling(15).std() * np.sqrt(365.25)
    df["eth_vol_30"] = eth_ret.rolling(30).std() * np.sqrt(365.25)
    
    # 8. BTC RSI (7, 14, 28 days)
    df["btc_rsi_7"] = compute_rsi(df["Close_btc"], 7)
    df["btc_rsi_14"] = compute_rsi(df["Close_btc"], 14)
    df["btc_rsi_28"] = compute_rsi(df["Close_btc"], 28)
    
    # 9. ETH RSI (7, 14, 28 days)
    df["eth_rsi_7"] = compute_rsi(df["Close_eth"], 7)
    df["eth_rsi_14"] = compute_rsi(df["Close_eth"], 14)
    df["eth_rsi_28"] = compute_rsi(df["Close_eth"], 28)
    
    # Target: Direction of spread change from t to t+1
    # 1 if spread_{t+1} > spread_t, else 0
    df["target"] = np.where(spread_series.shift(-1) > spread_series, 1.0, 0.0)
    
    # Drop rows that don't have full features
    df = df.dropna().reset_index(drop=True)
    return df

# PyTorch LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        # Take last timestep output
        last_out = out[:, -1, :] # (batch_size, hidden_dim)
        probs = self.sigmoid(self.fc(last_out))
        return probs

def create_sequences(features, targets, seq_len=30):
    xs, ys = [], []
    for i in range(len(features) - seq_len):
        xs.append(features[i:i+seq_len])
        # Target corresponds to the direction at the end of the sequence
        ys.append(targets[i+seq_len-1])
    return np.array(xs), np.array(ys)

def train_model(model, train_loader, epochs=15, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds.squeeze(1), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(train_loader.dataset)
        # print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    return model

def predict_model(model, test_loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            preds = model(batch_x)
            all_preds.extend(preds.squeeze(1).cpu().numpy())
    return np.array(all_preds)

def main():
    df = load_data()
    df = prepare_features(df)
    
    # Define features
    feature_cols = [
        "spread_mean_5", "spread_mean_15", "spread_mean_30",
        "spread_vol_5", "spread_vol_15", "spread_vol_30",
        "spread_rsi_7", "spread_rsi_14", "spread_rsi_28",
        "spread_roc_3", "spread_roc_10", "spread_roc_30",
        "spread_z_10", "spread_z_30", "spread_z_60",
        "btc_vol_5", "btc_vol_15", "btc_vol_30",
        "eth_vol_5", "eth_vol_15", "eth_vol_30",
        "btc_rsi_7", "btc_rsi_14", "btc_rsi_28",
        "eth_rsi_7", "eth_rsi_14", "eth_rsi_28"
    ]
    print(f"[*] Total features: {len(feature_cols)}")
    
    # Filter target dates (from Feb 2020)
    # We keep data from Jan 2018 for history/indicators, but training targets start Feb 2020
    eval_start_date = pd.to_datetime("2020-02-01")
    
    # Walk-forward setup
    # Fold 1: Train Feb 2020 - Dec 2022, Test Jan 2023 - Dec 2023
    # Fold 2: Train Feb 2020 - Dec 2023, Test Jan 2024 - Dec 2024
    # Fold 3: Train Feb 2020 - Dec 2024, Test Jan 2025 - May 2026 (or end of data)
    
    folds = [
        ("2020-02-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2020-02-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2020-02-01", "2024-12-31", "2025-01-01", "2026-05-31")
    ]
    
    seq_len = 30
    all_test_predictions = []
    
    for idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        print(f"\n[*] Training Fold {idx+1}: Train [{train_start} to {train_end}], Test [{test_start} to {test_end}]")
        
        # Convert strings to datetime
        ts_train_start = pd.to_datetime(train_start)
        ts_train_end = pd.to_datetime(train_end)
        ts_test_start = pd.to_datetime(test_start)
        ts_test_end = pd.to_datetime(test_end)
        
        # We need train sequences. Since sequences are 30 days long, we need history before train_start.
        # To get clean train_start targets, we select data from train_start - 30 days.
        train_df = df[(df["Date"] >= ts_train_start - pd.Timedelta(days=45)) & (df["Date"] <= ts_train_end)].copy()
        
        # Filter test set (needs 30 days lookback to predict on test_start)
        test_df = df[(df["Date"] >= ts_test_start - pd.Timedelta(days=45)) & (df["Date"] <= ts_test_end)].copy()
        
        # Purging: remove training sequences that overlap with the test period
        # Since sequence length is 30, the last 30 days of training have targets/data overlapping the test set.
        # So we restrict train_df targets to train_end - 30 days
        train_df_purged = train_df[train_df["Date"] <= ts_train_end - pd.Timedelta(days=seq_len)].copy()
        
        # Scale features
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_df_purged[feature_cols].values)
        test_features_scaled = scaler.transform(test_df[feature_cols].values)
        
        train_targets = train_df_purged["target"].values
        test_targets = test_df["target"].values
        
        # Create sequences
        X_train, y_train = create_sequences(train_features_scaled, train_targets, seq_len=seq_len)
        X_test, y_test = create_sequences(test_features_scaled, test_targets, seq_len=seq_len)
        
        # Get dates for test set predictions
        # The first prediction in X_test is at index seq_len in test_df
        test_dates = test_df["Date"].iloc[seq_len:].values
        test_spreads = test_df["spread"].iloc[seq_len:].values
        test_z_scores = test_df["z_score"].iloc[seq_len:].values
        test_betas = test_df["beta"].iloc[seq_len:].values
        test_alphas = test_df["alpha"].iloc[seq_len:].values
        test_closes_btc = test_df["Close_btc"].iloc[seq_len:].values
        test_closes_eth = test_df["Close_eth"].iloc[seq_len:].values
        test_opens_btc = test_df["Open_btc"].iloc[seq_len:].values
        test_opens_eth = test_df["Open_eth"].iloc[seq_len:].values
        
        # PyTorch loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Define and train model
        input_dim = len(feature_cols)
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=32, num_layers=2, dropout=0.2)
        model = train_model(model, train_loader, epochs=15, lr=0.001)
        
        # Predict
        preds_prob = predict_model(model, test_loader)
        
        # Store predictions
        fold_results = pd.DataFrame({
            "Date": test_dates,
            "Spread": test_spreads,
            "Z_Score": test_z_scores,
            "Beta": test_betas,
            "Alpha": test_alphas,
            "Close_BTC": test_closes_btc,
            "Close_ETH": test_closes_eth,
            "Open_BTC": test_opens_btc,
            "Open_ETH": test_opens_eth,
            "True_Direction": y_test,
            "Pred_Prob": preds_prob
        })
        # Filter to exact test date range (excluding lookback buffer)
        fold_results = fold_results[(fold_results["Date"] >= ts_test_start) & (fold_results["Date"] <= ts_test_end)]
        all_test_predictions.append(fold_results)
        print(f"[+] Fold {idx+1} complete. Generated {len(fold_results)} daily predictions.")
        
    # Concatenate all walk-forward predictions
    out_of_sample_df = pd.concat(all_test_predictions, ignore_index=True)
    out_of_sample_df["Pred_Dir"] = np.where(out_of_sample_df["Pred_Prob"] > 0.5, 1.0, 0.0)
    
    # Save predictions
    output_path = os.path.join(DATA_DIR, "lstm_predictions.csv")
    out_of_sample_df.to_csv(output_path, index=False)
    print(f"\n[+] Out-of-sample predictions successfully saved to {output_path}")
    print(f"    - Date Range: {out_of_sample_df['Date'].min()} to {out_of_sample_df['Date'].max()}")
    print(f"    - Total Rows: {len(out_of_sample_df)}")

if __name__ == "__main__":
    main()
