"""EDA 03: 相関分析・ラグ付きクロスコリレーション"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

load_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

# --- ピアソン相関行列ヒートマップ ---
fig, ax = plt.subplots(figsize=(8, 7))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Correlation Matrix (All Variables)")
fig.tight_layout()
fig.savefig(FIGURES_DIR + "correlation_heatmap.png")
plt.close()
print(f"Saved: {FIGURES_DIR}correlation_heatmap.png")

# --- OTと各負荷変数のラグ付き相関（±24時間） ---
max_lag = 24
lags = range(-max_lag, max_lag + 1)

fig, axes = plt.subplots(2, 3, figsize=(12, 10))
for ax, col in zip(axes.flatten(), load_cols):
    ccf_vals = []
    for lag in lags:
        if lag >= 0:
            c = (
                df["OT"]
                .iloc[lag:]
                .reset_index(drop=True)
                .corr(df[col].iloc[: len(df) - lag].reset_index(drop=True))
            )
        else:
            c = (
                df["OT"]
                .iloc[: len(df) + lag]
                .reset_index(drop=True)
                .corr(df[col].iloc[-lag:].reset_index(drop=True))
            )
        ccf_vals.append(c)
    ax.bar(list(lags), ccf_vals, width=1.0, color="tab:blue", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"OT × {col}")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Correlation")

fig.suptitle("Cross-Correlation: OT vs Load Variables (±24h)", fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR + "cross_correlation.png")
plt.close()
print(f"Saved: {FIGURES_DIR}cross_correlation.png")
