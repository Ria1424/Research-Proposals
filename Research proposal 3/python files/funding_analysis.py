import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    path = os.path.join(DATA_DIR, "btc_funding_daily.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}. Run data_fetcher.py first.")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def run_event_study(df):
    print("[*] Running event study on extreme funding rate readings...")
    
    # Calculate daily simple forward returns over 1d, 3d, and 7d horizons
    df["fwd_ret_1d"] = df["Close"].shift(-1) / df["Close"] - 1.0
    df["fwd_ret_3d"] = df["Close"].shift(-3) / df["Close"] - 1.0
    df["fwd_ret_7d"] = df["Close"].shift(-7) / df["Close"] - 1.0
    
    # Define extreme funding rate thresholds based on static 5th and 95th percentiles
    low_thresh = df["funding_rate"].quantile(0.05)
    high_thresh = df["funding_rate"].quantile(0.95)
    
    print(f"    - Static 5th percentile (Crowded Shorts): {low_thresh:.6f}")
    print(f"    - Static 95th percentile (Crowded Longs): {high_thresh:.6f}")
    
    events_shorts = df[df["funding_rate"] <= low_thresh].copy()
    events_longs = df[df["funding_rate"] >= high_thresh].copy()
    
    results = []
    
    # Helper to calculate statistics
    def calc_stats(event_df, name, direction_label):
        for horizon in ["1d", "3d", "7d"]:
            ret_col = f"fwd_ret_{horizon}"
            rets = event_df[ret_col].dropna().values
            n_events = len(rets)
            
            if n_events > 1:
                mean_ret = np.mean(rets)
                std_ret = np.std(rets, ddof=1)
                t_stat, p_val = ttest_1samp(rets, 0.0)
            else:
                mean_ret, std_ret, t_stat, p_val = 0.0, 0.0, 0.0, 1.0
                
            results.append({
                "Event Group": name,
                "Mechanism": direction_label,
                "Horizon": horizon,
                "Events Count": n_events,
                "Mean Return [%]": mean_ret * 100,
                "Std Dev [%]": std_ret * 100,
                "t-statistic": t_stat,
                "p-value": p_val
            })
            
    calc_stats(events_shorts, "Crowded Shorts (Low Funding <= 5%)", "Contrarian Long (+)")
    calc_stats(events_longs, "Crowded Longs (High Funding >= 95%)", "Contrarian Short (-)")
    
    event_study_df = pd.DataFrame(results)
    event_study_path = os.path.join(DATA_DIR, "event_study_summary.csv")
    event_study_df.to_csv(event_study_path, index=False)
    print(f"[+] Event study statistics completed and saved to {event_study_path}")
    print(event_study_df.to_string(index=False))
    return low_thresh, high_thresh

def simulate_perpetual_strategy(df, signals, include_funding=True, initial_capital=100000.0, fee_pct=0.001, slippage_pct=0.0005):
    """
    Simulates a strategy holding a perpetual position with daily rebalancing and funding payments.
    - signals: pandas Series of target weights at bar t, applied at Open of t+1.
      Weights: 1 (Long), -1 (Short), 0 (Flat)
    """
    # Shift signals by 1 to execute on the Open of the next day (no look-ahead)
    exec_signals = signals.shift(1).fillna(0.0).values
    
    dates = df["Date"].values
    opens = df["Open"].values
    closes = df["Close"].values
    funding_rates = df["funding_rate"].values
    
    cash = float(initial_capital)
    position_units = 0.0
    current_side = 0.0  # 1.0 = Long, -1.0 = Short, 0.0 = Flat
    
    equity_net = np.zeros(len(df))
    equity_gross = np.zeros(len(df))
    
    # Gross simulation variables (no fees, no slippage, no funding)
    cash_gross = float(initial_capital)
    pos_units_gross = 0.0
    current_side_gross = 0.0
    
    trades_count = 0
    win_count = 0
    loss_count = 0
    trade_entry_value = 0.0
    
    for i in range(len(df)):
        target_side = exec_signals[i]
        current_open = opens[i]
        current_close = closes[i]
        today_funding = funding_rates[i]
        
        # Evaluate current portfolio value at current Open (before rebalancing)
        open_equity = cash + position_units * current_open
        open_equity_gross = cash_gross + pos_units_gross * current_open
        
        # 1. Exit previous net position
        if target_side != current_side:
            if current_side != 0.0:
                exit_price = current_open * (1 - slippage_pct) if current_side == 1.0 else current_open * (1 + slippage_pct)
                if current_side == 1.0:
                    revenue = position_units * exit_price
                else:
                    revenue = -abs(position_units) * exit_price  # Pay to buy back
                    
                trade_value = abs(position_units) * exit_price
                fee = trade_value * fee_pct
                cash += revenue - fee
                
                # PnL tracking for hit rate
                exit_value = abs(position_units) * exit_price
                pnl = (exit_value - trade_entry_value) if current_side == 1.0 else (trade_entry_value - exit_value)
                if pnl > 0:
                    win_count += 1
                else:
                    loss_count += 1
                
                position_units = 0.0
                current_side = 0.0
                
            # Enter new position
            if target_side != 0.0:
                approx_fee_rate = fee_pct
                allocation = open_equity / (1.0 + approx_fee_rate)
                entry_price = current_open * (1 + slippage_pct) if target_side == 1.0 else current_open * (1 - slippage_pct)
                position_units = (allocation / entry_price) * target_side
                
                trade_value = abs(position_units) * entry_price
                fee = trade_value * fee_pct
                cash -= (trade_value if target_side == 1.0 else -trade_value) + fee
                
                trade_entry_value = abs(position_units) * entry_price
                current_side = target_side
                trades_count += 1
                
        # 2. Exit/Enter gross position
        if target_side != current_side_gross:
            if current_side_gross != 0.0:
                revenue_gross = pos_units_gross * current_open if current_side_gross == 1.0 else -abs(pos_units_gross) * current_open
                cash_gross += revenue_gross
                pos_units_gross = 0.0
                current_side_gross = 0.0
            if target_side != 0.0:
                pos_units_gross = (open_equity_gross / current_open) * target_side
                cash_gross -= (open_equity_gross if target_side == 1.0 else -open_equity_gross)
                current_side_gross = target_side
                
        # 3. Daily Funding Fee Application at Close
        funding_payment = 0.0
        if include_funding and current_side != 0.0:
            pos_value_close = abs(position_units) * current_close
            funding_payment = -current_side * today_funding * pos_value_close
            cash += funding_payment
            
        # 4. Value portfolio at Close
        equity_net[i] = cash + position_units * current_close
        equity_gross[i] = cash_gross + pos_units_gross * current_close
        
    hit_rate = (win_count / (win_count + loss_count) * 100) if (win_count + loss_count) > 0 else 0.0
    return equity_net, equity_gross, trades_count, hit_rate

def compute_metrics(equity, equity_gross, dates, trades_count, hit_rate, strategy_name):
    eq = pd.Series(equity)
    eq_gross = pd.Series(equity_gross)
    
    daily_ret = eq.pct_change().dropna()
    ann_factor = 365.0
    
    start_date = pd.to_datetime(dates[0])
    end_date = pd.to_datetime(dates[-1])
    years = (end_date - start_date).days / 365.0
    if years == 0:
        years = len(eq) / ann_factor
        
    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1.0
    total_return_gross = (eq_gross.iloc[-1] / eq_gross.iloc[0]) - 1.0
    cost_drag = total_return_gross - total_return
    
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 and eq.iloc[-1] > 0 else 0.0
    vol = daily_ret.std() * np.sqrt(ann_factor) if len(daily_ret) > 1 else 0.0
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(ann_factor)) if len(daily_ret) > 1 and daily_ret.std() > 0 else 0.0
    
    downside_ret = daily_ret[daily_ret < 0]
    sortino = (daily_ret.mean() / downside_ret.std() * np.sqrt(ann_factor)) if len(daily_ret) > 1 and len(downside_ret) > 1 and downside_ret.std() > 0 else 0.0
    
    peaks = eq.cummax()
    drawdowns = (eq - peaks) / peaks
    max_dd = drawdowns.min()
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
    
    return {
        "Strategy": strategy_name,
        "Total Return [%]": total_return * 100,
        "Total Return Gross [%]": total_return_gross * 100,
        "Cost Drag [%]": cost_drag * 100,
        "Annualized CAGR [%]": cagr * 100,
        "Annualized Volatility [%]": vol * 100,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown [%]": max_dd * 100,
        "Calmar Ratio": calmar,
        "Trade Count": trades_count,
        "Hit Rate [%]": hit_rate
    }

def main():
    df = load_data()
    
    # We start backtesting on day 200 to allow indicators (e.g. SMA 200) to warm up
    warmup = 200
    backtest_df = df.iloc[warmup:].reset_index(drop=True)
    dates = backtest_df["Date"].values
    
    # 1. Run event study (returns static thresholds)
    low_t, high_t = run_event_study(df)
    
    # 2. Simulate baseline 1: Buy and Hold BTC (Spot - no funding costs)
    print("\n[*] Simulating Buy-and-Hold BTC baseline...")
    bah_signals = pd.Series(1.0, index=range(len(backtest_df)))
    bah_eq, bah_eq_gross, bah_trades, bah_hit = simulate_perpetual_strategy(backtest_df, bah_signals, include_funding=False)
    bah_metrics = compute_metrics(bah_eq, bah_eq_gross, dates, bah_trades, bah_hit, "Buy and Hold BTC")
    
    # 3. Simulate baseline 2: SMA Crossover (50/200) (Spot - no funding costs)
    print("[*] Simulating SMA Crossover (50/200) baseline...")
    # Calculate SMA signals on the entire df first, then slice
    sma_50 = df["Close"].rolling(50).mean()
    sma_200 = df["Close"].rolling(200).mean()
    sma_signals_all = np.where(sma_50 > sma_200, 1.0, 0.0)
    sma_signals = pd.Series(sma_signals_all[warmup:], index=range(len(backtest_df)))
    
    sma_eq, sma_eq_gross, sma_trades, sma_hit = simulate_perpetual_strategy(backtest_df, sma_signals, include_funding=False)
    sma_metrics = compute_metrics(sma_eq, sma_eq_gross, dates, sma_trades, sma_hit, "SMA Crossover (50/200)")
    
    # 4. Simulate rule-based Contrarian Funding Rate Strategy (Perpetual - includes funding payments)
    print("[*] Simulating Rule-Based Contrarian Funding Rate strategy...")
    # Rolling 90-day 5th and 95th percentiles of funding rate on entire df, then slice
    roll_5pct = df["funding_rate"].rolling(90).quantile(0.05)
    roll_95pct = df["funding_rate"].rolling(90).quantile(0.95)
    
    # Contrarian Signals:
    # Go Long tomorrow (+1) if today's funding < 5th percentile (crowded shorts)
    # Go Short tomorrow (-1) if today's funding > 95th percentile (crowded longs)
    # Otherwise Flat (0)
    sig_array = np.zeros(len(df))
    for t in range(90, len(df)):
        f_rate = df["funding_rate"].iloc[t]
        l_p = roll_5pct.iloc[t]
        h_p = roll_95pct.iloc[t]
        
        if f_rate <= l_p:
            sig_array[t] = 1.0
        elif f_rate >= h_p:
            sig_array[t] = -1.0
        else:
            sig_array[t] = 0.0
            
    contrarian_signals = pd.Series(sig_array[warmup:], index=range(len(backtest_df)))
    
    contr_eq, contr_eq_gross, contr_trades, contr_hit = simulate_perpetual_strategy(backtest_df, contrarian_signals, include_funding=True)
    contr_metrics = compute_metrics(contr_eq, contr_eq_gross, dates, contr_trades, contr_hit, "Contrarian Funding Strategy")
    
    # 5. Plotting results
    print("[*] Generating comparison plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    
    colors = ["#777777", "#1f77b4", "#2ca02c"] # Grey, Blue, Green
    strategies = [
        (bah_eq, "Buy and Hold BTC", colors[0]),
        (sma_eq, "SMA Crossover (50/200)", colors[1]),
        (contr_eq, "Contrarian Funding Strategy", colors[2])
    ]
    
    # Equity curves
    for eq, name, col in strategies:
        ax1.plot(dates, eq, label=name, color=col, linewidth=1.5)
    ax1.set_title("Equity Curves: Contrarian Funding Strategy vs. Baselines (July 2020 - May 2026)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()
    
    # Drawdowns
    def get_dd_series(eq):
        peaks = pd.Series(eq).cummax()
        return (pd.Series(eq) - peaks) / peaks * 100
        
    for eq, name, col in strategies:
        dd = get_dd_series(eq)
        ax2.fill_between(dates, dd, 0, color=col, alpha=0.12)
        ax2.plot(dates, dd, label=name, color=col, linewidth=1.0)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "proposal3_performance.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[+] Performance plot saved to {plot_path}")
    
    # Save metrics summary
    metrics_summary_df = pd.DataFrame([bah_metrics, sma_metrics, contr_metrics])
    metrics_summary_df = metrics_summary_df.transpose()
    # Set headers
    metrics_summary_df.columns = metrics_summary_df.iloc[0]
    metrics_summary_df = metrics_summary_df.drop(metrics_summary_df.index[0])
    metrics_summary_df = metrics_summary_df.reset_index().rename(columns={"index": "Metric"})
    
    summary_path = os.path.join(DATA_DIR, "proposal3_metrics_summary.csv")
    metrics_summary_df.to_csv(summary_path, index=False)
    print(f"[+] Metrics summary saved to {summary_path}")
    print(metrics_summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
