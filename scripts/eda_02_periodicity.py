"""EDA 02: 周期性分析・STL分解・分散比"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL

# --- 設定 ---
DATA_PATH = "data/ETTh1.csv"
FIGURES_DIR = "outputs/figures/"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# --- データ読み込み ---
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df.set_index("date", inplace=True)

# --- 月別平均 ---
fig, ax = plt.subplots(figsize=(12, 6))
df.groupby(df.index.month)["OT"].mean().plot.bar(ax=ax, color="tab:red", alpha=0.8)
ax.set_title("OT Monthly Average")
ax.set_xlabel("Month")
ax.set_ylabel("OT")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(FIGURES_DIR + "ot_monthly_avg.png")
plt.close()
print(f"Saved: {FIGURES_DIR}ot_monthly_avg.png")

# --- 曜日別平均 ---
fig, ax = plt.subplots(figsize=(12, 6))
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_avg = df.groupby(df.index.dayofweek)["OT"].mean()
dow_avg.index = dow_labels
dow_avg.plot.bar(ax=ax, color="tab:blue", alpha=0.8)
ax.set_title("OT Day-of-Week Average")
ax.set_xlabel("Day of Week")
ax.set_ylabel("OT")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(FIGURES_DIR + "ot_weekday_avg.png")
plt.close()
print(f"Saved: {FIGURES_DIR}ot_weekday_avg.png")

# --- 時間帯別平均 ---
fig, ax = plt.subplots(figsize=(12, 6))
df.groupby(df.index.hour)["OT"].mean().plot.bar(ax=ax, color="tab:green", alpha=0.8)
ax.set_title("OT Hourly Average")
ax.set_xlabel("Hour")
ax.set_ylabel("OT")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
fig.tight_layout()
fig.savefig(FIGURES_DIR + "ot_hourly_avg.png")
plt.close()
print(f"Saved: {FIGURES_DIR}ot_hourly_avg.png")


# --- STL分解 ---
def plot_stl(result, period_label, save_path):
    """STL分解結果を4パネルで描画"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    components = [
        ("Observed", result.observed),
        ("Trend", result.trend),
        ("Seasonal", result.seasonal),
        ("Residual", result.resid),
    ]
    for ax, (title, data) in zip(axes, components):
        ax.plot(data, linewidth=0.5)
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(title)
    fig.suptitle(
        f"STL Decomposition of OT (period={period_label})", fontsize=16, y=1.01
    )
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


# 日次STL (period=24)
stl_daily = STL(df["OT"], period=24, robust=True).fit()
plot_stl(stl_daily, "24h", FIGURES_DIR + "ot_stl_daily.png")

# 週次STL (period=168)
stl_weekly = STL(df["OT"], period=168, robust=True).fit()
plot_stl(stl_weekly, "168h", FIGURES_DIR + "ot_stl_weekly.png")

# --- Seasonal成分ズーム: 最初の1週間（168時間） ---
zoom_end = df.index[0] + pd.Timedelta(hours=167)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for ax, (label, res) in zip(
    axes, [("Daily (24h)", stl_daily), ("Weekly (168h)", stl_weekly)]
):
    s = res.seasonal.loc[:zoom_end]
    ax.plot(s.index, s.values, linewidth=1.5, marker="o", markersize=3)
    ax.set_title(f"Seasonal Component — {label} (First 168h)", fontsize=14)
    ax.set_ylabel("Seasonal")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
fig.autofmt_xdate(rotation=30)
fig.tight_layout()
fig.savefig(FIGURES_DIR + "ot_seasonal_zoom.png")
plt.close()
print(f"Saved: {FIGURES_DIR}ot_seasonal_zoom.png")

# --- 分散比 ---
var_observed = df["OT"].var()

print()
for label, res in [("Daily (24h)", stl_daily), ("Weekly (168h)", stl_weekly)]:
    var_trend = res.trend.var()
    var_seasonal = res.seasonal.var()
    var_resid = res.resid.var()
    print(f"=== Variance Ratio — STL {label} ===")
    print(f"  Trend    / Observed = {var_trend / var_observed:.4f}  ({var_trend:.2f})")
    print(
        f"  Seasonal / Observed = {var_seasonal / var_observed:.4f}  ({var_seasonal:.2f})"
    )
    print(
        f"  Residual / Observed = {var_resid / var_observed:.4f}  ({var_resid:.2f})"
    )
    print()
