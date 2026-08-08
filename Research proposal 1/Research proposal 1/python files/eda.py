"""
eda.py
======
Full Exploratory Data Analysis for the crypto derivatives dataset.
Produces all plots and tables requested by professor:
  - Return distributions (histogram + QQ plot)
  - Autocorrelation (ACF/PACF)
  - Volatility clustering (rolling vol + GARCH-style vol-of-vol)
  - Regime plot (log price with shaded regimes)
  - Rolling correlation heatmap (BTC/ETH + macro)
  - Volume statistics
  - Missing value audit
  - Return heatmap (monthly returns calendar)

Outputs saved to results/eda/

Author: Ria Chawak | IIT Bombay Research Internship 2026
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (for server/script use)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path

EDA_DIR = Path("results/eda")
EDA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = Path("data/raw")

COLORS = {"BTC": "#F7931A", "ETH": "#627EEA", "BNB": "#F3BA2F", "neutral": "#4A90D9"}


def load_data():
    """Load all available instruments."""
    data = {}
    name_map = {"BTC": "BTC_USDT_ohlcv.parquet", "ETH": "ETH_USDT_ohlcv.parquet"}
    fr_map = {"BTC": "BTC_USDT_funding.parquet", "ETH": "ETH_USDT_funding.parquet"}

    for name, fname in name_map.items():
        path = RAW_DIR / fname
        if path.exists():
            df = pd.read_parquet(path)
            fr_path = RAW_DIR / fr_map[name]
            if fr_path.exists():
                fr = pd.read_parquet(fr_path).squeeze()
                df["funding_rate"] = fr.reindex(df.index).fillna(0)
            df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
            df["pct_ret"] = df["close"].pct_change()
            data[name] = df
    return data


def plot_return_distribution(data: dict):
    """Histogram + QQ plot for daily log returns."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Daily Log Return Distributions", fontsize=14, fontweight="bold")

    for idx, (name, df) in enumerate(data.items()):
        ret = df["log_ret"].dropna()
        ax_hist = axes[0, idx]
        ax_qq = axes[1, idx]

        # Histogram
        ax_hist.hist(ret, bins=100, color=COLORS.get(name, "#4A90D9"),
                     alpha=0.7, density=True, label="Empirical")
        x = np.linspace(ret.min(), ret.max(), 300)
        normal_pdf = stats.norm.pdf(x, ret.mean(), ret.std())
        ax_hist.plot(x, normal_pdf, "r--", linewidth=2, label="Normal fit")
        ax_hist.set_title(f"{name} Return Distribution")
        ax_hist.set_xlabel("Log Return")
        ax_hist.set_ylabel("Density")
        ax_hist.legend()

        # Stats annotation
        kurt = stats.kurtosis(ret)
        skew = stats.skew(ret)
        jb_stat, jb_p = stats.jarque_bera(ret)
        ax_hist.text(0.05, 0.95,
                     f"Kurtosis: {kurt:.2f}\nSkewness: {skew:.2f}\nJB p-value: {jb_p:.3e}",
                     transform=ax_hist.transAxes, va="top", fontsize=9,
                     bbox=dict(facecolor="white", alpha=0.8))

        # QQ plot
        stats.probplot(ret, dist="norm", plot=ax_qq)
        ax_qq.set_title(f"{name} QQ Plot (vs Normal)")
        ax_qq.get_lines()[1].set_color("red")

    plt.tight_layout()
    fig.savefig(EDA_DIR / "return_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: return_distributions.png")


def plot_autocorrelation(data: dict):
    """ACF and PACF of returns and squared returns."""
    for name, df in data.items():
        ret = df["log_ret"].dropna()
        sq_ret = ret ** 2

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f"{name} Autocorrelation Analysis", fontsize=13, fontweight="bold")

        plot_acf(ret, ax=axes[0, 0], lags=20, alpha=0.05, color=COLORS.get(name))
        axes[0, 0].set_title("ACF — Daily Returns (look for momentum/reversion)")

        plot_pacf(ret, ax=axes[0, 1], lags=20, alpha=0.05, color=COLORS.get(name))
        axes[0, 1].set_title("PACF — Daily Returns")

        plot_acf(sq_ret, ax=axes[1, 0], lags=20, alpha=0.05, color="orange")
        axes[1, 0].set_title("ACF — Squared Returns (GARCH/volatility clustering)")

        plot_pacf(sq_ret, ax=axes[1, 1], lags=20, alpha=0.05, color="orange")
        axes[1, 1].set_title("PACF — Squared Returns")

        plt.tight_layout()
        fig.savefig(EDA_DIR / f"autocorrelation_{name.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: autocorrelation_{name.lower()}.png")

        # Print lag-1 autocorrelation
        lag1_ret = ret.autocorr(lag=1)
        lag1_sq = sq_ret.autocorr(lag=1)
        print(f"  {name} lag-1 ACF (returns): {lag1_ret:.4f}")
        print(f"  {name} lag-1 ACF (squared returns): {lag1_sq:.4f}")


def plot_regime(data: dict):
    """Log price chart with shaded market regimes."""
    REGIMES = [
        ("2020-01-01", "2020-02-15", "grey",   "Early 2020 Accumulation"),
        ("2020-02-15", "2020-03-20", "red",    "COVID Crash"),
        ("2020-03-20", "2021-04-15", "green",  "2020-21 Bull Run"),
        ("2021-04-15", "2021-07-20", "orange", "Mid-2021 Correction"),
        ("2021-07-20", "2021-11-10", "green",  "2021 Bull Leg 2"),
        ("2021-11-10", "2022-05-10", "red",    "Bear Start"),
        ("2022-05-10", "2022-06-30", "darkred","Terra/Luna Collapse"),
        ("2022-07-01", "2022-11-05", "orange", "Bear Recovery Attempt"),
        ("2022-11-05", "2022-12-31", "darkred","FTX Collapse"),
        ("2023-01-01", "2024-01-10", "green",  "2023 Recovery"),
        ("2024-01-10", "2026-06-01", "green",  "ETF / 2024-26 Bull"),
    ]

    fig, axes = plt.subplots(len(data), 1, figsize=(16, 6 * len(data)))
    if len(data) == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, data.items()):
        log_close = np.log(df["close"])
        ax.plot(df.index, log_close, color=COLORS.get(name), linewidth=1.2, zorder=5)

        for start, end, color, label in REGIMES:
            s = pd.Timestamp(start, tz="UTC")
            e = pd.Timestamp(end, tz="UTC")
            if s > df.index[-1] or e < df.index[0]:
                continue
            ax.axvspan(max(s, df.index[0]), min(e, df.index[-1]),
                       alpha=0.15, color=color, zorder=1)

        ax.set_title(f"{name} — Log Price with Market Regimes", fontsize=12, fontweight="bold")
        ax.set_ylabel("Log Price (USD)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(alpha=0.3)

        # Overlay rolling 20d realized vol (scaled for visual)
        rv = df["log_ret"].rolling(20).std() * np.sqrt(365)
        rv_scaled = log_close.mean() + (rv - rv.mean()) / rv.std() * log_close.std() * 0.5
        ax.plot(df.index, rv_scaled, color="black", linewidth=0.8, alpha=0.5,
                linestyle="--", label="Realized Vol (scaled)")
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(EDA_DIR / "regime_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: regime_plot.png")


def plot_rolling_correlation(data: dict):
    """Rolling 90-day correlation between BTC and ETH."""
    if "BTC" not in data or "ETH" not in data:
        return
    btc = data["BTC"]["log_ret"]
    eth = data["ETH"]["log_ret"]
    common_idx = btc.dropna().index.intersection(eth.dropna().index)
    btc_c = btc.loc[common_idx]
    eth_c = eth.loc[common_idx]

    rolling_corr = btc_c.rolling(90).corr(eth_c)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("BTC/ETH Rolling Correlation & Return Scatter", fontsize=13, fontweight="bold")

    ax1.plot(rolling_corr.index, rolling_corr, color="#2E5099", linewidth=1.2)
    ax1.axhline(rolling_corr.mean(), color="red", linestyle="--", alpha=0.7, label=f"Mean: {rolling_corr.mean():.2f}")
    ax1.fill_between(rolling_corr.index, rolling_corr, rolling_corr.mean(),
                     where=rolling_corr < rolling_corr.mean(), alpha=0.2, color="red",
                     label="Below avg (pairs opportunity)")
    ax1.set_title("90-Day Rolling Correlation: BTC vs ETH Daily Returns")
    ax1.set_ylabel("Correlation")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1)

    # Scatter plot
    ax2.scatter(btc_c, eth_c, alpha=0.2, s=5, color="#4A90D9")
    slope, intercept, r, p, se = stats.linregress(btc_c, eth_c)
    x_line = np.array([btc_c.min(), btc_c.max()])
    ax2.plot(x_line, slope * x_line + intercept, "r-", linewidth=2,
             label=f"OLS fit (β={slope:.2f}, R²={r**2:.3f})")
    ax2.set_title("BTC vs ETH Daily Log Returns (Full Sample)")
    ax2.set_xlabel("BTC Log Return")
    ax2.set_ylabel("ETH Log Return")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(EDA_DIR / "btc_eth_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: btc_eth_correlation.png")
    print(f"  BTC/ETH full-sample correlation: {btc_c.corr(eth_c):.3f}")
    print(f"  Rolling 90d correlation range: {rolling_corr.min():.2f} – {rolling_corr.max():.2f}")


def plot_monthly_heatmap(data: dict):
    """Monthly returns calendar heatmap."""
    import matplotlib.colors as mcolors
    for name, df in data.items():
        monthly = df["close"].resample("ME").last().pct_change()
        monthly.index = monthly.index.to_period("M")
        pivot = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
        pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]

        fig, ax = plt.subplots(figsize=(14, len(pivot) * 0.7 + 1))
        cmap = plt.cm.RdYlGn
        vmax = min(pivot.abs().max().max(), 0.5)
        im = ax.imshow(pivot.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label="Monthly Return")

        ax.set_xticks(range(12))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index.tolist())

        for i in range(len(pivot)):
            for j in range(12):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=7, color="black" if abs(val) < 0.3 else "white")

        ax.set_title(f"{name} — Monthly Returns Heatmap", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fig.savefig(EDA_DIR / f"monthly_heatmap_{name.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: monthly_heatmap_{name.lower()}.png")


def compute_volume_stats(data: dict):
    """Print and save volume statistics."""
    rows = []
    for name, df in data.items():
        v = df["volume"]
        rows.append({
            "Instrument": name,
            "Mean Daily Volume": f"{v.mean():.0f}",
            "Median Daily Volume": f"{v.median():.0f}",
            "Std Daily Volume": f"{v.std():.0f}",
            "Max Daily Volume": f"{v.max():.0f}",
            "Skewness": f"{stats.skew(v):.2f}",
            "Days >2x 30d avg": str(((v / v.rolling(30).mean()) > 2).sum()),
        })
    vol_df = pd.DataFrame(rows)
    vol_df.to_csv(EDA_DIR / "volume_stats.csv", index=False)
    print("\nVolume Statistics:")
    print(vol_df.to_string(index=False))
    return vol_df


def missing_value_audit(data: dict):
    """Check and report missing values per instrument."""
    print("\nMissing Value Audit:")
    for name, df in data.items():
        n_missing = df.isnull().sum()
        missing_nonzero = n_missing[n_missing > 0]
        if len(missing_nonzero) > 0:
            print(f"  {name}: MISSING VALUES FOUND:\n{missing_nonzero}")
        else:
            print(f"  {name}: No missing values. Shape: {df.shape}")

        # Date gap check
        date_range = pd.date_range(df.index[0], df.index[-1], freq="D", tz="UTC")
        missing_dates = date_range.difference(df.index)
        if len(missing_dates) > 0:
            print(f"  {name}: {len(missing_dates)} missing dates: {list(missing_dates[:3])}")
        else:
            print(f"  {name}: No date gaps found.")


def descriptive_stats(data: dict):
    """Print detailed descriptive statistics for each instrument."""
    all_stats = []
    for name, df in data.items():
        ret = df["log_ret"].dropna()
        s = {
            "Instrument": name,
            "N Bars": len(df),
            "Date Range": f"{df.index[0].date()} → {df.index[-1].date()}",
            "Mean Daily Return": f"{ret.mean():.4%}",
            "Std Daily Return": f"{ret.std():.4%}",
            "Ann. Volatility": f"{ret.std() * np.sqrt(365):.1%}",
            "Skewness": f"{stats.skew(ret):.3f}",
            "Kurtosis (excess)": f"{stats.kurtosis(ret):.3f}",
            "JB p-value": f"{stats.jarque_bera(ret)[1]:.2e}",
            "Max 1-day Drop": f"{ret.min():.2%}",
            "Max 1-day Gain": f"{ret.max():.2%}",
        }
        all_stats.append(s)
        print(f"\n  {name} Descriptive Stats:")
        for k, v in s.items():
            print(f"    {k:<30s} {v}")

    pd.DataFrame(all_stats).to_csv(EDA_DIR / "descriptive_stats.csv", index=False)


if __name__ == "__main__":
    print("Loading data...")
    data = load_data()
    if not data:
        print("No data found. Run data_loader.py first.")
        exit()

    print(f"Loaded: {list(data.keys())}\n")

    print("Running EDA...\n")
    missing_value_audit(data)
    descriptive_stats(data)
    compute_volume_stats(data)

    print("\nGenerating plots...")
    plot_return_distribution(data)
    plot_autocorrelation(data)
    plot_regime(data)
    plot_rolling_correlation(data)
    plot_monthly_heatmap(data)

    print(f"\nAll EDA outputs saved to {EDA_DIR}/")
    print("Files:")
    for f in sorted(EDA_DIR.iterdir()):
        print(f"  {f.name}")
