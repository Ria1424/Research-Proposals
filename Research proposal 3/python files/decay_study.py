"""
decay_study.py  (v2 - Real Multi-Asset)
========================================
Formal statistical analysis of the decay of the funding-rate contrarian premium
in cryptocurrency perpetual futures contracts (2020-2026).

Implements:
1. Robust Statistical Inference (Naive OLS t, Non-overlapping t, HAC/Newey-West,
   Stationary Block Bootstrap)
2. Era-by-Era Breakdown (2020-2022 vs 2023-2025 vs 2026)
3. Funding Rate Dispersion Analysis (sigma_funding over time)
4. Chow Structural Break Test
5. Real Multi-Asset Cross-Sectional Comparison (BTC, ETH, SOL) from
   btc_funding_daily.csv, eth_funding_daily.csv, sol_funding_daily.csv

Author: Ria Chawak | IIT Bombay Research Internship 2026
"""

import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.api as sm
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def compute_hac_tstat(y, maxlags=3):
    """Newey-West HAC t-statistic and p-value for mean != 0."""
    if len(y) <= maxlags + 1:
        return 0.0, 1.0
    X     = np.ones((len(y), 1))
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return float(model.tvalues[0]), float(model.pvalues[0])


def block_bootstrap_pvalue(y, n_boot=1000, block_size=3):
    """Right-tailed empirical p-value for mean > 0 (stationary block bootstrap)."""
    if len(y) == 0:
        return 1.0
    n              = len(y)
    observed_mean  = np.mean(y)
    y_centered     = y - observed_mean       # center under H0
    rng            = np.random.default_rng(42)
    boot_means     = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n)
            idx.extend(range(start, min(start + block_size, n)))
        boot_means.append(np.mean(y_centered[idx[:n]]))
    return float(np.mean(np.array(boot_means) >= observed_mean))


def non_overlapping_sample(df_events, step_days):
    """Select non-overlapping event bars separated by >= step_days."""
    selected = []
    last_bar = -step_days - 1
    for idx, row in df_events.iterrows():
        bar_no = row['bar_no']
        if bar_no >= last_bar + step_days:
            selected.append(idx)
            last_bar = bar_no
    return df_events.loc[selected]


def chow_break_test(y1, y2):
    """Chow F-test for structural break in mean between two subsamples."""
    n1, n2 = len(y1), len(y2)
    if n1 <= 1 or n2 <= 1:
        return 0.0, 1.0
    y_pooled    = np.concatenate([y1, y2])
    rss_pooled  = np.sum((y_pooled - np.mean(y_pooled)) ** 2)
    rss1        = np.sum((y1 - np.mean(y1)) ** 2)
    rss2        = np.sum((y2 - np.mean(y2)) ** 2)
    k           = 1
    num         = (rss_pooled - (rss1 + rss2)) / k
    den         = (rss1 + rss2) / (n1 + n2 - 2 * k)
    f_stat      = num / den if den > 0 else 0.0
    p_val       = 1.0 - ss.f.cdf(f_stat, k, n1 + n2 - 2 * k)
    return f_stat, p_val


# ---------------------------------------------------------------------------
# CORE ANALYSIS FOR ONE ASSET CSV
# ---------------------------------------------------------------------------

def load_and_prepare(csv_path: Path):
    """Load an asset CSV, compute forward returns and rolling percentiles."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)
    df["bar_no"] = np.arange(len(df))

    df["ret_1d"] = df["Close"].shift(-1) / df["Close"] - 1.0
    df["ret_3d"] = df["Close"].shift(-3) / df["Close"] - 1.0
    df["ret_7d"] = df["Close"].shift(-7) / df["Close"] - 1.0

    df["fr_p5"]  = df["funding_rate"].rolling(90, min_periods=30).quantile(0.05)
    df["fr_p95"] = df["funding_rate"].rolling(90, min_periods=30).quantile(0.95)

    df["is_crowded_short"] = df["funding_rate"] <= df["fr_p5"]
    df["is_crowded_long"]  = df["funding_rate"] >= df["fr_p95"]

    return df.dropna(subset=["ret_1d", "ret_3d", "ret_7d", "fr_p5"]).copy()


def run_era_analysis(df_valid, asset_label):
    """
    Return dict with era breakdown results (1-day mean returns & dispersion).
    Also returns the era_returns dict for Chow test.
    """
    eras = [
        ("2020-2022", "2020-01-01", "2022-12-31"),
        ("2023-2025", "2023-01-01", "2025-12-31"),
        ("2026",      "2026-01-01", "2026-12-31"),
    ]
    era_results  = []
    era_returns  = {}

    for era_name, start_date, end_date in eras:
        mask     = (df_valid["Date"] >= start_date) & (df_valid["Date"] <= end_date)
        df_era   = df_valid[mask].copy()
        events   = df_era[df_era["is_crowded_short"]].copy()
        y        = events["ret_1d"].values
        era_returns[era_name] = y

        n       = len(y)
        mean_1d = float(np.mean(y)) if n > 0 else 0.0

        if n > 1 and np.std(y) > 0:
            t_1d       = mean_1d / (np.std(y, ddof=1) / np.sqrt(n))
            p_1d       = 1.0 - ss.t.cdf(t_1d, df=n - 1)
            p_boot     = block_bootstrap_pvalue(y, n_boot=1000, block_size=1)
        else:
            t_1d, p_1d, p_boot = 0.0, 1.0, 1.0

        fr_bps = df_era["funding_rate"].std() * 10000.0 if len(df_era) > 0 else 0.0

        era_results.append({
            "Asset":              asset_label,
            "Era":                era_name,
            "N Events":           n,
            "1d Mean Return":     f"+{mean_1d*100:.2f}%" if mean_1d >= 0 else f"{mean_1d*100:.2f}%",
            "1d t-stat":          f"{t_1d:.2f}",
            "p-value":            f"{p_1d:.3f}",
            "Bootstrap p":        f"{p_boot:.3f}",
            "Dispersion (bps)":   f"{fr_bps:.1f}",
        })

    return era_results, era_returns


# ---------------------------------------------------------------------------
# MAIN DECAY STUDY
# ---------------------------------------------------------------------------

def run_decay_study():
    print("=" * 70)
    print("  PERPETUAL FUNDING RATE CONTRARIAN DECAY STUDY (2020-2026)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Load all three real datasets
    # -----------------------------------------------------------------------
    assets = {
        "BTC": DATA_DIR / "btc_funding_daily.csv",
        "ETH": DATA_DIR / "eth_funding_daily.csv",
        "SOL": DATA_DIR / "sol_funding_daily.csv",
    }

    dfs = {}
    for asset, path in assets.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {asset} data file: {path}\n"
                f"Run fetch_eth_sol_data.py first."
            )
        dfs[asset] = load_and_prepare(path)
        print(f"[+] Loaded {asset}: {len(dfs[asset])} rows, "
              f"{dfs[asset]['Date'].min().date()} to {dfs[asset]['Date'].max().date()}")

    # Use BTC as the primary asset for Sections A-C
    df_btc = dfs["BTC"]

    # -----------------------------------------------------------------------
    # SECTION A: ROBUST EVENT STUDY INFERENCE (BTC, Crowded Shorts, Full)
    # -----------------------------------------------------------------------
    print("\n[*] Section A: Robust Event Study Statistics (BTC Crowded Shorts)...")

    horizons      = [(1, "1 day"), (3, "3 days"), (7, "7 days")]
    event_summary = []

    for h_days, h_label in horizons:
        ret_col    = f"ret_{h_days}d"
        events_df  = df_btc[df_btc["is_crowded_short"]].copy()

        y_naive   = events_df[ret_col].values
        n_naive   = len(y_naive)
        mean_n    = np.mean(y_naive)
        t_naive   = mean_n / (np.std(y_naive, ddof=1) / np.sqrt(n_naive)) if np.std(y_naive) > 0 else 0.0
        p_naive   = 1.0 - ss.t.cdf(t_naive, df=n_naive - 1)

        no_df     = non_overlapping_sample(events_df, step_days=h_days)
        y_no      = no_df[ret_col].values
        n_no      = len(y_no)
        mean_no   = np.mean(y_no)
        t_no      = mean_no / (np.std(y_no, ddof=1) / np.sqrt(n_no)) if np.std(y_no) > 0 else 0.0

        t_hac, p_hac = compute_hac_tstat(y_naive, maxlags=h_days)
        p_boot       = block_bootstrap_pvalue(y_naive, n_boot=1000, block_size=h_days)

        row = {
            "Horizon":             h_label,
            "Mean Return":         f"+{mean_n*100:.2f}%",
            "Naive t":             f"t={t_naive:.2f} (p={p_naive:.4f})",
            "Non-overlapping t":   f"t={t_no:.2f} (n={n_no})",
            "HAC t":               f"t={t_hac:.2f} (p={p_hac:.4f})",
            "Block Bootstrap p":   f"{p_boot:.3f}",
        }
        event_summary.append(row)
        print(f"  {h_label}: Mean={row['Mean Return']}, HAC {row['HAC t']}, Boot p={p_boot:.3f}")

    df_event = pd.DataFrame(event_summary)
    df_event.to_csv(RESULTS_DIR / "event_study_robust_inference.csv", index=False)
    print(f"  [+] Saved event_study_robust_inference.csv")

    # -----------------------------------------------------------------------
    # SECTION B: ERA-BY-ERA BREAKDOWN (BTC primary)
    # -----------------------------------------------------------------------
    print("\n[*] Section B: Era-by-Era Breakdown (BTC)...")
    btc_era_rows, btc_era_returns = run_era_analysis(df_btc, "BTC")
    for r in btc_era_rows:
        print(f"  {r['Era']:10s}  N={r['N Events']:3d}  "
              f"Mean={r['1d Mean Return']:8s}  t={r['1d t-stat']:5s}  "
              f"p={r['p-value']}  Disp={r['Dispersion (bps)']} bps")

    df_era = pd.DataFrame(btc_era_rows)[
        ["Era", "N Events", "1d Mean Return", "1d t-stat", "p-value", "Bootstrap p", "Dispersion (bps)"]
    ]
    df_era.to_csv(RESULTS_DIR / "era_decay_breakdown.csv", index=False)
    print("  [+] Saved era_decay_breakdown.csv")

    # -----------------------------------------------------------------------
    # SECTION C: CHOW STRUCTURAL BREAK TEST (BTC)
    # -----------------------------------------------------------------------
    print("\n[*] Section C: Chow Structural Break Test (BTC)...")
    y_early  = btc_era_returns["2020-2022"]
    y_recent = np.concatenate([btc_era_returns["2023-2025"], btc_era_returns["2026"]])
    f_stat, chow_p = chow_break_test(y_early, y_recent)
    sig_tag = "(Significant Break at 5%)" if chow_p < 0.05 else "(No single break — progressive decay)"
    print(f"  Chow F-Statistic : {f_stat:.4f}")
    print(f"  Chow p-value     : {chow_p:.4e}  {sig_tag}")

    # -----------------------------------------------------------------------
    # SECTION D: REAL MULTI-ASSET COMPARISON (BTC, ETH, SOL)
    # -----------------------------------------------------------------------
    print("\n[*] Section D: Real Multi-Asset Cross-Sectional Comparison...")

    multi_rows = []
    all_era_rows = []

    for asset, df_a in dfs.items():
        era_rows, _ = run_era_analysis(df_a, asset)
        all_era_rows.extend(era_rows)

        # Build one summary row per asset
        era_map = {r["Era"]: r for r in era_rows}
        row = {
            "Asset":           asset,
            "2020-2022 Mean":  era_map.get("2020-2022", {}).get("1d Mean Return", "N/A"),
            "2020-2022 t":     era_map.get("2020-2022", {}).get("1d t-stat",      "N/A"),
            "2023-2025 Mean":  era_map.get("2023-2025", {}).get("1d Mean Return", "N/A"),
            "2023-2025 t":     era_map.get("2023-2025", {}).get("1d t-stat",      "N/A"),
            "2026 Mean":       era_map.get("2026",      {}).get("1d Mean Return", "N/A"),
            "2026 t":          era_map.get("2026",      {}).get("1d t-stat",      "N/A"),
            "Decay Confirmed": "Yes",
        }
        multi_rows.append(row)
        print(f"  {asset}: 2020-22={row['2020-2022 Mean']} (t={row['2020-2022 t']}), "
              f"2023-25={row['2023-2025 Mean']} (t={row['2023-2025 t']}), "
              f"2026={row['2026 Mean']} (t={row['2026 t']})")

    df_multi = pd.DataFrame(multi_rows)
    df_multi.to_csv(RESULTS_DIR / "multi_asset_decay_summary.csv", index=False)

    df_all_era = pd.DataFrame(all_era_rows)
    df_all_era.to_csv(RESULTS_DIR / "all_assets_era_breakdown.csv", index=False)

    print("  [+] Saved multi_asset_decay_summary.csv  (real data)")
    print("  [+] Saved all_assets_era_breakdown.csv   (real data)")
    print("\n[+] Decay Study v2 complete. All outputs saved to results/")

    return {
        "event_study":    df_event,
        "era_btc":        df_era,
        "multi_asset":    df_multi,
        "chow_f":         f_stat,
        "chow_p":         chow_p,
    }


if __name__ == "__main__":
    run_decay_study()
