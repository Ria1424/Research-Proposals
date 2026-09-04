"""
metrics_v2.py
=============
Standardized central metrics library for quantitative backtest reconciliation.
Ensures identical mathematical definitions for CAGR, Volatility, Sharpe,
Max Drawdown, and Deflated Sharpe Ratio (DSR) across all engines and proposals.
"""

import numpy as np
import pandas as pd
import scipy.stats as ss

def calculate_cagr(equity_series, n_years):
    """Calculate Annualized Compound Annual Growth Rate (CAGR)."""
    if n_years <= 0 or len(equity_series) == 0:
        return 0.0
    val_start = float(equity_series.iloc[0])
    val_end = float(equity_series.iloc[-1])
    if val_start <= 0 or val_end <= 0:
        return 0.0
    return (val_end / val_start) ** (1.0 / n_years) - 1.0

def calculate_volatility(equity_series, ann_factor=365.0, winsor_cap=0.50):
    """Calculate Annualized Volatility of daily returns.
    Winsorizes at +/-winsor_cap to suppress engine-artifact discontinuities."""
    if len(equity_series) <= 1:
        return 0.0
    returns = equity_series.pct_change().dropna()
    if len(returns) <= 1:
        return 0.0
    returns = returns.clip(lower=-winsor_cap, upper=winsor_cap)
    return float(returns.std(ddof=1) * np.sqrt(ann_factor))

def calculate_sharpe_ratio(cagr, vol):
    """Calculate Sharpe Ratio (excess return over risk-free rate of 0%)."""
    if vol <= 0:
        return 0.0
    return float(cagr / vol)

def calculate_max_drawdown(equity_series):
    """Calculate Maximum Drawdown percentage."""
    if len(equity_series) == 0:
        return 0.0
    roll_max = equity_series.cummax()
    drawdowns = (equity_series - roll_max) / roll_max
    return float(drawdowns.min())

def calculate_deflated_sharpe_ratio(returns, n_trials=100, ann_factor=365.0):
    """
    Calculate Deflated Sharpe Ratio (DSR) to adjust for multi-test selection bias
    and non-normal return distributions.
    """
    n = len(returns)
    if n <= 2:
        return 0.0
    mean = returns.mean()
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    sr_daily = mean / std
    
    skew = returns.skew()
    kurt = returns.kurtosis()
    if kurt is None or np.isnan(kurt):
        kurt = 0.0
    kurt = kurt + 3.0  # Convert to Pearson kurtosis
    
    sr_var = (1.0 - skew * sr_daily + (kurt - 1.0) / 4.0 * sr_daily**2) / (n - 1.0)
    sr_std = np.sqrt(max(sr_var, 1e-8))
    emc = 0.5772156649
    max_z = (1.0 - emc) * ss.norm.ppf(1.0 - 1.0 / n_trials) + emc * ss.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr_benchmark_daily = max_z * sr_std
    
    dsr = ss.norm.cdf((sr_daily - sr_benchmark_daily) / sr_std)
    return float(dsr)

def compute_standard_metrics(equity_series, n_years, n_trials=100, ann_factor=365.0):
    """
    Computes all standard quantitative metrics for an equity curve.
    """
    equity_series = pd.Series(equity_series)
    cagr = calculate_cagr(equity_series, n_years)
    vol = calculate_volatility(equity_series, ann_factor)
    sharpe = calculate_sharpe_ratio(cagr, vol)
    max_dd = calculate_max_drawdown(equity_series)
    
    returns = equity_series.pct_change().dropna()
    dsr = calculate_deflated_sharpe_ratio(returns, n_trials, ann_factor)
    
    return {
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "dsr": dsr
    }
