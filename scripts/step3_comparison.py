"""Step 3: 全モデル比較（Baseline + ML + DL + Informer）"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 設定 ---
FIGURES_DIR = "outputs/figures/"
HORIZONS = [1, 6, 24, 168]

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

# --- 結果集約 ---
# Step 1: ベースライン
baseline_results = {
    "Persistence": {"1h": 0.4353, "6h": 1.1636, "24h": 1.6324, "168h": 2.7617},
    "SeasonalNaive24": {"1h": 1.6302, "6h": 1.6295, "24h": 1.6324, "168h": 1.6389},
}

# Step 2: ML Direct
ml_results = {
    "Ridge": {"1h": 0.6673, "6h": 1.2674, "24h": 1.9343, "168h": 2.7513},
    "LightGBM": {"1h": 0.7905, "6h": 1.5614, "24h": 3.1435, "168h": 6.3294},
}

# Step 3: DL（CSVから読み込み）
lstm_df = pd.read_csv("outputs/lstm_results.csv")
patchtst_df = pd.read_csv("outputs/patchtst_results.csv")

dl_results = {"LSTM": {}, "PatchTST": {}}
for _, row in lstm_df.iterrows():
    dl_results["LSTM"][row["Horizon"]] = row["MAE"]
for _, row in patchtst_df.iterrows():
    dl_results["PatchTST"][row["Horizon"]] = row["MAE"]

# RMSE
rmse_lstm = {}
rmse_patchtst = {}
for _, row in lstm_df.iterrows():
    rmse_lstm[row["Horizon"]] = row["RMSE"]
for _, row in patchtst_df.iterrows():
    rmse_patchtst[row["Horizon"]] = row["RMSE"]

baseline_rmse = {
    "Persistence": {"1h": 0.6298, "6h": 1.5552, "24h": 2.1344, "168h": 3.5603},
    "SeasonalNaive24": {"1h": 2.1320, "6h": 2.1317, "24h": 2.1344, "168h": 2.1432},
}
ml_rmse = {
    "Ridge": {"1h": 0.9079, "6h": 1.6765, "24h": 2.5138, "168h": 3.4063},
    "LightGBM": {"1h": 1.0521, "6h": 2.0192, "24h": 3.9392, "168h": 7.0369},
}

# Step 3.5: Informer（npyから計算）
# 24hモデル: step1=1h, step6=6h, step24(last)=24h
pred24 = np.load("results/informer_ETTh1_ftS_sl168_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_poc_eval_0/pred.npy")
true24 = np.load("results/informer_ETTh1_ftS_sl168_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_poc_eval_0/true.npy")
# 168hモデル: step168(last)=168h
pred168 = np.load("results/informer_ETTh1_ftS_sl336_ll168_pl168_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_poc_eval_0/pred.npy")
true168 = np.load("results/informer_ETTh1_ftS_sl336_ll168_pl168_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_poc_eval_0/true.npy")

def mae_at_step(pred, true, step_idx):
    return np.mean(np.abs(pred[:, step_idx, 0] - true[:, step_idx, 0]))

def rmse_at_step(pred, true, step_idx):
    return np.sqrt(np.mean((pred[:, step_idx, 0] - true[:, step_idx, 0])**2))

informer_mae = {
    "1h": mae_at_step(pred24, true24, 0),
    "6h": mae_at_step(pred24, true24, 5),
    "24h": mae_at_step(pred24, true24, -1),
    "168h": mae_at_step(pred168, true168, -1),
}
informer_rmse = {
    "1h": rmse_at_step(pred24, true24, 0),
    "6h": rmse_at_step(pred24, true24, 5),
    "24h": rmse_at_step(pred24, true24, -1),
    "168h": rmse_at_step(pred168, true168, -1),
}

# --- 全結果テーブル ---
all_models = ["Persistence", "SeasonalNaive24", "Ridge", "LightGBM", "LSTM", "PatchTST", "Informer"]

def get_mae(model, hkey):
    if model in baseline_results:
        return baseline_results[model][hkey]
    elif model in ml_results:
        return ml_results[model][hkey]
    elif model in dl_results:
        return dl_results[model].get(hkey, float("nan"))
    elif model == "Informer":
        return informer_mae[hkey]
    return float("nan")

def get_rmse(model, hkey):
    if model in baseline_rmse:
        return baseline_rmse[model][hkey]
    elif model in ml_rmse:
        return ml_rmse[model][hkey]
    elif model == "LSTM":
        return rmse_lstm.get(hkey, float("nan"))
    elif model == "PatchTST":
        return rmse_patchtst.get(hkey, float("nan"))
    elif model == "Informer":
        return informer_rmse[hkey]
    return float("nan")

print("=== Full Model Comparison (MAE / RMSE) ===")
print(f"{'Model':20s}", end="")
for h in HORIZONS:
    print(f"  {h}h MAE    {h}h RMSE ", end="")
print()
print("-" * 120)

for model in all_models:
    print(f"{model:20s}", end="")
    for h in HORIZONS:
        hkey = f"{h}h"
        mae_val = get_mae(model, hkey)
        rmse_val = get_rmse(model, hkey)
        print(f"  {mae_val:7.4f}  {rmse_val:7.4f} ", end="")
    print()

# --- 改善率テーブル ---
print("\n=== Improvement vs Best Baseline (MAE %) ===")
print(f"{'Model':20s}", end="")
for h in HORIZONS:
    print(f"  {h:>4}h", end="")
print()
print("-" * 60)

for model in all_models:
    print(f"{model:20s}", end="")
    for h in HORIZONS:
        hkey = f"{h}h"
        best_baseline = min(baseline_results["Persistence"][hkey],
                           baseline_results["SeasonalNaive24"][hkey])
        mae_val = get_mae(model, hkey)
        improvement = (best_baseline - mae_val) / best_baseline * 100
        sign = "+" if improvement > 0 else ""
        print(f"  {sign}{improvement:5.1f}%", end="")
    print()

# --- 可視化1: 全モデルMAE棒グラフ ---
colors = {
    "Persistence": "gray", "SeasonalNaive24": "silver",
    "Ridge": "tab:purple", "LightGBM": "tab:green",
    "LSTM": "tab:red", "PatchTST": "tab:blue",
    "Informer": "tab:orange",
}

x = np.arange(len(HORIZONS))
width = 0.11

fig, ax = plt.subplots(figsize=(14, 6))
for i, model in enumerate(all_models):
    vals = [get_mae(model, f"{h}h") for h in HORIZONS]
    ax.bar(x + i * width, vals, width, label=model, color=colors[model], alpha=0.85)

ax.set_title("MAE Comparison: All Models (incl. Informer)")
ax.set_xlabel("Horizon")
ax.set_ylabel("MAE")
ax.set_xticks(x + width * (len(all_models) - 1) / 2)
ax.set_xticklabels([f"{h}h" for h in HORIZONS])
ax.legend(fontsize=10, ncol=3)
fig.tight_layout()
fig.savefig(FIGURES_DIR + "all_models_with_informer.png")
plt.close()
print(f"\nSaved: {FIGURES_DIR}all_models_with_informer.png")

# --- 可視化2: Informer 24h予測 vs 実測（最初2週間） ---
DATA_PATH = "data/ETTh1.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df.set_index("date", inplace=True)

# Informerのテストデータ開始位置: border1 = 12*30*24+4*30*24 - seq_len = 11520-168 = 11352
# 予測対象: t+1 ~ t+24 なので、最初の予測の最終ステップは index 11352+168+24-1 = 11543
# ただし pred24 の各サンプルはsliding windowなので、i番目のサンプルの最終予測は index (11352+i)+168+24-1
# 簡易的に最終ステップ(24h先)の予測をプロット

n_plot = 14 * 24
informer_test_start = 12*30*24 + 4*30*24  # 11520
n = min(n_plot, len(pred24))
plot_start = informer_test_start + 24 - 1  # 最初の予測の24h先の位置
plot_index = df.index[plot_start : plot_start + n]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(plot_index, true24[:n, -1, 0], label="Actual", color="black", linewidth=1.2)
ax.plot(plot_index, pred24[:n, -1, 0], label="Informer", color="tab:orange", linewidth=0.9, alpha=0.9)
ax.set_title("Informer Prediction vs Actual (Horizon=24h, First 2 Weeks)")
ax.set_xlabel("Date")
ax.set_ylabel("OT")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR + "informer_prediction_24h.png")
plt.close()
print(f"Saved: {FIGURES_DIR}informer_prediction_24h.png")

# --- 可視化3: 短期ホライズンのみ比較（1h, 6hに絞った拡大図） ---
short_horizons = [1, 6]
short_models = ["Persistence", "Ridge", "LSTM", "Informer"]

x2 = np.arange(len(short_horizons))
width2 = 0.18

fig, ax = plt.subplots(figsize=(8, 5))
for i, model in enumerate(short_models):
    vals = [get_mae(model, f"{h}h") for h in short_horizons]
    ax.bar(x2 + i * width2, vals, width2, label=model, color=colors[model], alpha=0.85)

ax.set_title("MAE Comparison: Short-term Horizons (1h, 6h)")
ax.set_xlabel("Horizon")
ax.set_ylabel("MAE")
ax.set_xticks(x2 + width2 * (len(short_models) - 1) / 2)
ax.set_xticklabels([f"{h}h" for h in short_horizons])
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR + "short_horizon_comparison.png")
plt.close()
print(f"Saved: {FIGURES_DIR}short_horizon_comparison.png")
