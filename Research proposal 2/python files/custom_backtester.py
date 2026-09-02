import os
import pandas as pd
import numpy as np

class CustomBacktester:
    def __init__(self, df, initial_capital=100000.0, market_type="crypto", slippage_pct=0.0005):
        """
        Parameters:
        - df: pandas DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        - initial_capital: Starting capital (float)
        - market_type: 'crypto' or 'indian_equities'
        - slippage_pct: Percentage slippage to apply to execution price
        """
        self.df = df.copy()
        # Sort and clean
        self.df = self.df.sort_values("Date").reset_index(drop=True)
        self.initial_capital = float(initial_capital)
        self.market_type = market_type
        self.slippage_pct = float(slippage_pct)

    def calculate_indian_fees(self, trade_value, is_buy):
        """
        Calculates Indian market frictions for delivery equities:
        - Brokerage: 0.03% capped at Rs. 20 per trade
        - STT (Securities Transaction Tax): 0.1% on both Buy and Sell
        - Exchange Transaction Charges (NSE): 0.00345%
        - SEBI Turnover Fee: 0.0001% (Rs. 10 per crore)
        - GST: 18% of (Brokerage + Exchange Charges + SEBI fee)
        - Stamp Duty: 0.015% on Buy only
        """
        trade_value = float(trade_value)
        brokerage = min(0.0003 * trade_value, 20.0)
        stt = 0.001 * trade_value
        exchange_charges = 0.0000345 * trade_value
        sebi_fee = 0.000001 * trade_value
        gst = 0.18 * (brokerage + exchange_charges + sebi_fee)
        stamp_duty = 0.00015 * trade_value if is_buy else 0.0
        
        total_fees = brokerage + stt + exchange_charges + sebi_fee + gst + stamp_duty
        return total_fees

    def calculate_crypto_fees(self, trade_value):
        """
        Calculates standard crypto exchange fees:
        - 0.1% taker fee (e.g., Binance standard fee)
        """
        return 0.001 * float(trade_value)

    def run_simulation(self, signals, long_only=True):
        """
        Runs the simulation.
        - signals: list/array of target weights at bar t.
          Must be aligned with df index.
          Weights can be 1 (fully long), 0 (flat/cash), or -1 (fully short if long_only=False).
        - long_only: If True, short signals (-1) are treated as flat (0).
        """
        # Ensure signals is a series and shift by 1 for execution next day
        signals_series = pd.Series(signals)
        exec_signals = signals_series.shift(1).fillna(0).values
        
        # Initialize net portfolio variables
        cash = self.initial_capital
        position = 0.0  # units of asset held
        
        # Initialize gross portfolio variables (zero fees/slippage for drag calculation)
        cash_gross = self.initial_capital
        position_gross = 0.0
        
        dates = self.df["Date"].values
        opens = self.df["Open"].values
        closes = self.df["Close"].values
        
        equity_curve = np.zeros(len(self.df))
        equity_curve_gross = np.zeros(len(self.df))
        
        trades_log = []
        
        for i in range(len(self.df)):
            current_open = opens[i]
            current_close = closes[i]
            target_weight = exec_signals[i]
            
            if long_only and target_weight < 0:
                target_weight = 0.0
                
            # 1. Current portfolio value evaluated at current Open
            current_equity = cash + position * current_open
            current_equity_gross = cash_gross + position_gross * current_open
            
            # Current weights before rebalancing
            current_weight = (position * current_open) / current_equity if current_equity > 0 else 0.0
            current_weight_gross = (position_gross * current_open) / current_equity_gross if current_equity_gross > 0 else 0.0
            
            # Rebalance NET Portfolio
            if target_weight != current_weight:
                target_value = target_weight * current_equity
                current_value = position * current_open
                trade_value = target_value - current_value
                
                if trade_value > 0:  # BUY
                    buy_price = current_open * (1 + self.slippage_pct)
                    # Sizing logic considering fees
                    approx_fee_rate = 0.001 if self.market_type == "crypto" else 0.0015
                    max_buy_value = cash / (1 + approx_fee_rate)
                    actual_trade_value = min(trade_value, max_buy_value)
                    
                    if actual_trade_value > 0:
                        shares_to_buy = actual_trade_value / buy_price
                        gross_cost = shares_to_buy * buy_price
                        
                        if self.market_type == "crypto":
                            fees = self.calculate_crypto_fees(gross_cost)
                        else:
                            fees = self.calculate_indian_fees(gross_cost, is_buy=True)
                            
                        # Adjust if we overshoot cash
                        if cash - (gross_cost + fees) < 0:
                            scale = cash / (gross_cost + fees)
                            shares_to_buy *= scale * 0.999
                            gross_cost = shares_to_buy * buy_price
                            if self.market_type == "crypto":
                                fees = self.calculate_crypto_fees(gross_cost)
                            else:
                                fees = self.calculate_indian_fees(gross_cost, is_buy=True)
                                
                        cash = cash - gross_cost - fees
                        position += shares_to_buy
                        
                        trades_log.append({
                            "Date": dates[i],
                            "Type": "BUY",
                            "Price": buy_price,
                            "Units": shares_to_buy,
                            "Value": gross_cost,
                            "Fees": fees,
                            "Cash_After": cash
                        })
                        
                elif trade_value < 0:  # SELL
                    sell_price = current_open * (1 - self.slippage_pct)
                    shares_to_sell = abs(trade_value) / sell_price
                    
                    # If long-only, cannot sell more than we own
                    if long_only:
                        shares_to_sell = min(shares_to_sell, position)
                        
                    if shares_to_sell > 0:
                        gross_revenue = shares_to_sell * sell_price
                        if self.market_type == "crypto":
                            fees = self.calculate_crypto_fees(gross_revenue)
                        else:
                            fees = self.calculate_indian_fees(gross_revenue, is_buy=False)
                            
                        cash = cash + gross_revenue - fees
                        position -= shares_to_sell
                        
                        trades_log.append({
                            "Date": dates[i],
                            "Type": "SELL",
                            "Price": sell_price,
                            "Units": shares_to_sell,
                            "Value": gross_revenue,
                            "Fees": fees,
                            "Cash_After": cash
                        })
                        
            # Rebalance GROSS Portfolio (no fees, no slippage)
            if target_weight != current_weight_gross:
                target_value_gross = target_weight * current_equity_gross
                current_value_gross = position_gross * current_open
                trade_value_gross = target_value_gross - current_value_gross
                
                if trade_value_gross > 0:  # BUY
                    shares_to_buy = trade_value_gross / current_open
                    cash_gross -= shares_to_buy * current_open
                    position_gross += shares_to_buy
                elif trade_value_gross < 0:  # SELL
                    shares_to_sell = abs(trade_value_gross) / current_open
                    if long_only:
                        shares_to_sell = min(shares_to_sell, position_gross)
                    cash_gross += shares_to_sell * current_open
                    position_gross -= shares_to_sell
                    
            # 3. Value portfolio at close of bar
            equity_curve[i] = cash + position * current_close
            equity_curve_gross[i] = cash_gross + position_gross * current_close
            
        return equity_curve, equity_curve_gross, trades_log

    def compute_metrics(self, equity_curve, equity_curve_gross, trades):
        """
        Computes key quantitative performance metrics.
        """
        eq = pd.Series(equity_curve)
        eq_gross = pd.Series(equity_curve_gross)
        
        # Daily Returns
        daily_ret = eq.pct_change().dropna()
        
        # General parameters
        n_bars = len(eq)
        is_crypto = (self.market_type == "crypto")
        ann_factor = 365.25 if is_crypto else 252.0
        
        start_date = pd.to_datetime(self.df["Date"].iloc[0])
        end_date = pd.to_datetime(self.df["Date"].iloc[-1])
        years = (end_date - start_date).days / 365.25
        if years == 0:
            years = n_bars / ann_factor
            
        # Total Return
        total_return = (eq.iloc[-1] / eq.iloc[0]) - 1.0
        total_return_gross = (eq_gross.iloc[-1] / eq_gross.iloc[0]) - 1.0
        cost_drag = total_return_gross - total_return
        
        # CAGR
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 and eq.iloc[-1] > 0 else 0.0
        
        # Volatility
        vol = daily_ret.std() * np.sqrt(ann_factor) if len(daily_ret) > 1 else 0.0
        
        # Sharpe Ratio
        rf = 0.0  # Assumed 0% for simple excess return
        sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(ann_factor)) if len(daily_ret) > 1 and daily_ret.std() > 0 else 0.0
        
        # Sortino Ratio
        downside_ret = daily_ret[daily_ret < 0]
        sortino = (daily_ret.mean() / downside_ret.std() * np.sqrt(ann_factor)) if len(daily_ret) > 1 and len(downside_ret) > 1 and downside_ret.std() > 0 else 0.0
        
        # Drawdowns
        peaks = eq.cummax()
        drawdowns = (eq - peaks) / peaks
        max_dd = drawdowns.min()
        
        # Calmar Ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0
        
        # Hit Rate (Profitable Trades)
        trade_returns = []
        # Group buy and sell to count trades
        # Simplify: match every BUY to next SELL (or vice-versa) to calculate hit rate
        win_count = 0
        loss_count = 0
        
        # A simple matched trade calculator for hit rate:
        # Loop through trades and calculate return of matched round trips
        running_pos = 0.0
        buy_value = 0.0
        sell_value = 0.0
        
        for t in trades:
            if t["Type"] == "BUY":
                buy_value += t["Value"]
            else:
                sell_value += t["Value"]
                
        # Approximate: if sell_value > buy_value, profitable
        # Let's do a more precise trade matching
        entry_price = 0.0
        entry_units = 0.0
        for t in trades:
            if entry_units == 0:
                entry_price = t["Price"]
                entry_units = t["Units"]
                entry_type = t["Type"]
            else:
                # Close trade
                exit_price = t["Price"]
                exit_units = t["Units"]
                # Calculate return
                if entry_type == "BUY":
                    ret = (exit_price / entry_price) - 1.0
                else:
                    ret = (entry_price / exit_price) - 1.0
                
                if ret > 0:
                    win_count += 1
                else:
                    loss_count += 1
                entry_units = 0.0 # reset
                
        total_trades = win_count + loss_count
        hit_rate = win_count / total_trades if total_trades > 0 else 0.0
        
        metrics = {
            "Total Return [%]": total_return * 100,
            "Total Return Gross [%]": total_return_gross * 100,
            "Cost Drag [%]": cost_drag * 100,
            "Annualized CAGR [%]": cagr * 100,
            "Annualized Volatility [%]": vol * 100,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown [%]": max_dd * 100,
            "Calmar Ratio": calmar,
            "Trade Count": len(trades),
            "Hit Rate [%]": hit_rate * 100
        }
        return metrics

    def run_sanity_checks(self):
        """
        Validates the engine against the two baseline checks.
        """
        print("[*] Running BacktestEngine validation checks...")
        
        # 1. Buy and Hold Validation
        # Signal = 1 for all rows
        bh_signals = np.ones(len(self.df))
        eq_net, eq_gross, trades = self.run_simulation(bh_signals, long_only=True)
        
        bh_gross_return = (eq_gross[-1] / eq_gross[0]) - 1.0
        # Since signals are shifted by 1, the first trade happens at the Open of day 1 (index 1)
        asset_return = (self.df["Close"].iloc[-1] / self.df["Open"].iloc[1]) - 1.0
        
        # A true Buy and Hold gross return should match the asset's Open-to-Close return exactly
        bh_check = np.isclose(bh_gross_return, asset_return, rtol=1e-4)
        if bh_check:
            print("[+] Sanity Check 1 Passed: Buy & Hold gross return matches asset return exactly.")
        else:
            print(f"[!] Sanity Check 1 Failed: B&H Gross={bh_gross_return:.4%}, Asset={asset_return:.4%}")
            
        # 2. Random/Zero-Edge Strategy Cost Drag Validation
        # Signal random choice of 0 or 1 every 20 bars
        np.random.seed(42)
        rand_signals = np.zeros(len(self.df))
        curr_sig = 0
        for idx in range(len(self.df)):
            if idx % 20 == 0:
                curr_sig = np.random.choice([0, 1])
            rand_signals[idx] = curr_sig
            
        eq_net_r, eq_gross_r, trades_r = self.run_simulation(rand_signals, long_only=True)
        
        net_ret = (eq_net_r[-1] / eq_net_r[0]) - 1.0
        gross_ret = (eq_gross_r[-1] / eq_gross_r[0]) - 1.0
        
        print(f"    - Random Signals run: Gross Return = {gross_ret:.4%}, Net Return = {net_ret:.4%}")
        if net_ret < gross_ret:
            print(f"[+] Sanity Check 2 Passed: Random signals show positive cost drag (Net < Gross). Cost drag = {(gross_ret-net_ret):.4%}")
        else:
            print("[!] Sanity Check 2 Failed: No cost drag observed for random signals.")
            
        return bh_check, (net_ret < gross_ret)

if __name__ == "__main__":
    # Test script with dummy data if run directly
    print("[*] Initializing Custom Backtester Sanity Test...")
    dates = pd.date_range("2020-01-01", periods=100)
    np.random.seed(42)
    prices = 100.0 * np.cumprod(1 + np.random.normal(0, 0.01, 100))
    df_dummy = pd.DataFrame({
        "Date": dates.date,
        "Open": prices * 0.99,
        "High": prices * 1.01,
        "Low": prices * 0.98,
        "Close": prices,
        "Volume": np.random.randint(1000, 10000, 100)
    })
    
    tester = CustomBacktester(df_dummy, market_type="crypto")
    tester.run_sanity_checks()
