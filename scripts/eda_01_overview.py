"""EDA 01: データ概要・基本統計量・時系列プロット"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# --- 基本情報 ---
print("=== df.info() ===")
df.info()
print()

print("=== df.describe() ===")
print(df.describe().to_string())
print()

print("=== 欠損値 ===")
print(df.isnull().sum().to_string())
print(f"\n全体の欠損数: {df.isnull().sum().sum()}")
print(f"データ期間: {df.index.min()} ~ {df.index.max()}")
print(f"レコード数: {len(df)}")

# --- OT 全期間時系列プロット ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["OT"], linewidth=0.5, color="tab:red")
ax.set_title("OT (Oil Temperature) — Full Period")
ax.set_xlabel("Date")
ax.set_ylabel("OT")
fig.savefig(FIGURES_DIR + "ot_timeseries_full.png")
plt.close()
print(f"\nSaved: {FIGURES_DIR}ot_timeseries_full.png")

# --- 負荷変数サブプロット ---
load_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
for ax, col in zip(axes.flatten(), load_cols):
    ax.plot(df.index, df[col], linewidth=0.5)
    ax.set_title(col, fontsize=14)
    ax.set_ylabel(col)
fig.suptitle("Load Features — Full Period", fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR + "load_features_full.png")
plt.close()
print(f"Saved: {FIGURES_DIR}load_features_full.png")
