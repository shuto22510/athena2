"""Step 1: ベースラインモデル（Persistence / Seasonal Naive / Mean）
- SeasonalNaive24 は「直近24時間を未来に繰り返し貼る」定義（未来参照なし）
"""

# model_01_baseline.py — Persistence/SeasonalNaive/Meanの評価（リーク修正済み）
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 設定 ---
DATA_PATH = "data/ETTh1.csv"
FIGURES_DIR = "outputs/figures/"
HORIZONS = [1, 6, 24, 168]
PERIOD = 24  # 24h seasonality (ETTh1 is hourly)

os.makedirs(FIGURES_DIR, exist_ok=True)

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

# --- データ読み込み・分割 ---
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df.set_index("date", inplace=True)
ot = df["OT"].values

TRAIN_END = 12 * 30 * 24  # 8640
VAL_END = TRAIN_END + 4 * 30 * 24  # 11520

train_ot = ot[:TRAIN_END]
test_ot = ot[VAL_END:]
test_index = df.index[VAL_END:]

train_mean = train_ot.mean()

print(f"Train: {TRAIN_END} samples (index 0~{TRAIN_END - 1})")
print(f"Val:   {VAL_END - TRAIN_END} samples (index {TRAIN_END}~{VAL_END - 1})")
print(f"Test:  {len(test_ot)} samples (index {VAL_END}~{len(ot) - 1})")
print(f"Train OT mean: {train_mean:.2f}")
print()

# --- 評価関数 ---
def mae(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.mean(np.abs(y_true[mask] - y_pred[mask]))

def rmse(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))

# --- ベースライン予測生成・評価 ---
results = []

for h in HORIZONS:
    # 予測対象: y[t+h] を、時刻 t の情報だけで予測する
    # t の範囲: VAL_END .. len(ot)-h-1
    n_test = len(ot) - VAL_END - h
    if n_test <= 0:
        continue

    # 実績 y[t+h]
    actual = ot[VAL_END + h : VAL_END + h + n_test]

    # Persistence: yhat[t+h] = y[t]
    pred_persist = ot[VAL_END : VAL_END + n_test]

    # Seasonal Naive (24h, no leakage):
    # 「直近24時間の波形を未来に繰り返し貼る」
    # k は 1..24 のどれか（未来の時間帯）
    # k = ((h-1) % 24) + 1
    # yhat[t+h] = y[t - 24 + k]
    k = ((h - 1) % PERIOD) + 1
    start = VAL_END - PERIOD + k
    if start < 0:
        raise ValueError("Not enough history for SeasonalNaive24. Increase warmup or adjust split.")
    pred_seasonal = ot[start : start + n_test]

    # Mean: yhat = train_mean
    pred_mean = np.full(n_test, train_mean)

    for name, pred in [
        ("Persistence", pred_persist),
        ("SeasonalNaive24", pred_seasonal),
        ("Mean", pred_mean),
    ]:
        results.append({
            "Model": name,
            "Horizon": f"{h}h",
            "MAE": mae(actual, pred),
            "RMSE": rmse(actual, pred),
        })

# --- 結果テーブル ---
results_df = pd.DataFrame(results)
print("=== Baseline Results (Test Set) ===")
for h in HORIZONS:
    print(f"\n--- Horizon: {h}h ---")
    sub = results_df[results_df["Horizon"] == f"{h}h"]
    for _, row in sub.iterrows():
        print(f"  {row['Model']:20s}  MAE={row['MAE']:.4f}  RMSE={row['RMSE']:.4f}")

# --- 可視化1: テスト期間の予測 vs 実績（最初2週間ズーム, horizon=24h） ---
h_plot = 24
n_plot = 14 * 24  # 2週間
n_test_plot = min(n_plot, len(ot) - VAL_END - h_plot)

plot_index = test_index[h_plot : h_plot + n_test_plot]
actual_plot = ot[VAL_END + h_plot : VAL_END + h_plot + n_test_plot]
persist_plot = ot[VAL_END : VAL_END + n_test_plot]

k_plot = ((h_plot - 1) % PERIOD) + 1
start_plot = VAL_END - PERIOD + k_plot
seasonal_plot = ot[start_plot : start_plot + n_test_plot]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(plot_index, actual_plot, label="Actual", color="black", linewidth=1.2)
ax.plot(plot_index, persist_plot, label="Persistence", color="tab:blue", linewidth=0.8, alpha=0.8)
ax.plot(plot_index, seasonal_plot, label="Seasonal Naive (24h)", color="tab:orange", linewidth=0.8, alpha=0.8)
ax.axhline(train_mean, label="Mean", color="tab:red", linewidth=0.8, linestyle="--", alpha=0.7)
ax.set_title(f"Baseline Predictions vs Actual (Horizon={h_plot}h, First 2 Weeks)")
ax.set_xlabel("Date")
ax.set_ylabel("OT")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "baseline_predictions.png"))
plt.close()
print(f"\nSaved: {os.path.join(FIGURES_DIR, 'baseline_predictions.png')}")

# --- 可視化2: ホライズン別MAE/RMSEの棒グラフ ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

models = ["Persistence", "SeasonalNaive24", "Mean"]
colors = ["tab:blue", "tab:orange", "tab:red"]
x = np.arange(len(HORIZONS))
width = 0.25

for metric, ax in zip(["MAE", "RMSE"], axes):
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = []
        for h in HORIZONS:
            v = results_df[(results_df["Model"] == model) & (results_df["Horizon"] == f"{h}h")][metric].values
            vals.append(v[0] if len(v) else np.nan)
        ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.8)

    ax.set_title(metric)
    ax.set_xlabel("Horizon")
    ax.set_ylabel(metric)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{h}h" for h in HORIZONS])
    ax.legend()

fig.suptitle("Baseline Model Comparison by Horizon", fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "baseline_metrics.png"))
plt.close()
print(f"Saved: {os.path.join(FIGURES_DIR, 'baseline_metrics.png')}")
