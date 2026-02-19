"""EDA 04: 定常性検定・ACF/PACF"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

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

# --- ACF / PACF プロット（lag=168） ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_acf(df["OT"].dropna(), lags=168, ax=axes[0], alpha=0.05)
axes[0].set_title("ACF of OT (lag=168h)")
axes[0].set_xlabel("Lag (hours)")

plot_pacf(df["OT"].dropna(), lags=168, ax=axes[1], alpha=0.05, method="ywm")
axes[1].set_title("PACF of OT (lag=168h)")
axes[1].set_xlabel("Lag (hours)")

fig.tight_layout()
fig.savefig(FIGURES_DIR + "ot_acf_pacf.png")
plt.close()
print(f"Saved: {FIGURES_DIR}ot_acf_pacf.png")

# --- ADF検定 ---
adf_result = adfuller(df["OT"].dropna(), autolag="AIC")

print()
print("=== ADF Test on OT ===")
print(f"Test Statistic : {adf_result[0]:.6f}")
print(f"p-value        : {adf_result[1]:.6f}")
print(f"Lags Used      : {adf_result[2]}")
print(f"Observations   : {adf_result[3]}")
print("Critical Values:")
for key, val in adf_result[4].items():
    print(f"  {key}: {val:.6f}")
