"""スケール検証: Informer出力の値域確認 + 全モデル正規化スケール統一評価 + 可視化"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# データ読み込み
# ============================================================
df = pd.read_csv("data/ETTh1.csv")
ot = df["OT"].values

TRAIN_END = 12 * 30 * 24  # 8640
VAL_END = TRAIN_END + 4 * 30 * 24  # 11520

train_ot = ot[:TRAIN_END]
inf_mean = train_ot.mean()  # ddof=0 (numpy default = Informer方式)
inf_std = train_ot.std()    # ddof=0

pred24 = np.load("results/informer_ETTh1_ftS_sl168_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_poc_eval_0/pred.npy")
true24 = np.load("results/informer_ETTh1_ftS_sl168_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_poc_eval_0/true.npy")
pred168 = np.load("results/informer_ETTh1_ftS_sl336_ll168_pl168_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_poc_eval_0/pred.npy")
true168 = np.load("results/informer_ETTh1_ftS_sl336_ll168_pl168_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_poc_eval_0/true.npy")


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mse_f(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# ============================================================
# 1. Informer出力の値域確認
# ============================================================
print("=" * 70)
print("1. Informer出力（npy）の値域確認")
print("=" * 70)
print(f"pred24:  shape={pred24.shape}, min={pred24.min():.4f}, max={pred24.max():.4f}, mean={pred24.mean():.4f}")
print(f"true24:  shape={true24.shape}, min={true24.min():.4f}, max={true24.max():.4f}, mean={true24.mean():.4f}")
print(f"pred168: shape={pred168.shape}, min={pred168.min():.4f}, max={pred168.max():.4f}, mean={pred168.mean():.4f}")
print(f"true168: shape={true168.shape}, min={true168.min():.4f}, max={true168.max():.4f}, mean={true168.mean():.4f}")
print()
print(f"true24[0,:5,0]     = {true24[0,:5,0]}")
print(f"OT[11520:11525]    = {ot[11520:11525]}")
print(f"  → 完全一致 → true は元スケール（℃）")
print()
print(f"テスト期間のOT: min={ot[VAL_END:].min():.2f}, max={ot[VAL_END:].max():.2f}")
print(f"pred24 範囲:    min={pred24.min():.2f}, max={pred24.max():.2f}")
print(f"  → pred も元スケール（℃）の範囲")
print()
print(">>> 結論: Informerのnpyファイルは元スケール（℃）で保存されている")

# ============================================================
# 2. 学習セットのOT統計量
# ============================================================
print()
print("=" * 70)
print("2. 学習セットのOT統計量")
print("=" * 70)
print(f"Train OT (0~{TRAIN_END-1}): {TRAIN_END} samples")
print(f"  mean = {inf_mean:.6f}")
print(f"  std  = {inf_std:.6f}  (ddof=0, numpy default = Informer方式)")
print()
print(f"変換式:")
print(f"  正規化:   x_norm = (x - {inf_mean:.6f}) / {inf_std:.6f}")
print(f"  逆正規化: x_orig = x_norm * {inf_std:.6f} + {inf_mean:.6f}")

# ============================================================
# 3. 全モデルの結果 → 正規化スケールへ変換
# ============================================================
print()
print("=" * 70)
print("3. 正規化スケール変換の計算過程")
print("=" * 70)
print(f"変換: MAE_norm = MAE_orig / std = MAE_orig / {inf_std:.6f}")
print(f"変換: MSE_norm = MSE_orig / std^2 = MSE_orig / {inf_std**2:.6f}")

# 検算
print()
print("検算: Informer MAE を2通りで計算:")
for h, pred, true in [(24, pred24, true24), (168, pred168, true168)]:
    mae_orig = mae(true[:, -1, 0], pred[:, -1, 0])
    mae_converted = mae_orig / inf_std
    true_norm = (true[:, -1, 0] - inf_mean) / inf_std
    pred_norm = (pred[:, -1, 0] - inf_mean) / inf_std
    mae_direct = mae(true_norm, pred_norm)
    print(f"  {h}h: 変換値={mae_converted:.6f}, 直接計算={mae_direct:.6f}, 差={abs(mae_converted - mae_direct):.8f}")

# Informer 元スケール
informer_orig = {}
informer_orig[24] = {"MAE": mae(true24[:, -1, 0], pred24[:, -1, 0]),
                     "MSE": mse_f(true24[:, -1, 0], pred24[:, -1, 0])}
informer_orig[168] = {"MAE": mae(true168[:, -1, 0], pred168[:, -1, 0]),
                      "MSE": mse_f(true168[:, -1, 0], pred168[:, -1, 0])}

# Step1-3 元スケール結果
step_results = {
    "Persistence":     {24: {"MAE": 1.6324, "RMSE": 2.1344}, 168: {"MAE": 2.7617, "RMSE": 3.5603}},
    "SeasonalNaive24": {24: {"MAE": 1.6324, "RMSE": 2.1344}, 168: {"MAE": 1.6389, "RMSE": 2.1432}},
    "Ridge":           {24: {"MAE": 1.9343, "RMSE": 2.5138}, 168: {"MAE": 2.7513, "RMSE": 3.4063}},
    "LightGBM":        {24: {"MAE": 3.1435, "RMSE": 3.9392}, 168: {"MAE": 6.3294, "RMSE": 7.0369}},
    "LSTM":            {24: {"MAE": 2.7225, "RMSE": 3.3540}, 168: {"MAE": 5.8927, "RMSE": 6.6080}},
    "PatchTST":        {24: {"MAE": 6.1443, "RMSE": 7.0426}, 168: {"MAE": 6.6102, "RMSE": 7.3197}},
}

# 正規化スケールに変換
norm_results = {}
for model, horizons in step_results.items():
    norm_results[model] = {}
    for h, vals in horizons.items():
        norm_results[model][h] = {
            "MAE": vals["MAE"] / inf_std,
            "MSE": vals["RMSE"] ** 2 / inf_std ** 2,
        }

norm_results["Informer(自分)"] = {}
for h in [24, 168]:
    norm_results["Informer(自分)"][h] = {
        "MAE": informer_orig[h]["MAE"] / inf_std,
        "MSE": informer_orig[h]["MSE"] / inf_std ** 2,
    }

paper = {24: {"MAE": 0.247, "MSE": 0.098}, 168: {"MAE": 0.346, "MSE": 0.183}}

# ============================================================
# 4. 統一比較テーブル
# ============================================================
all_models = [
    "Persistence", "SeasonalNaive24",
    "Ridge", "LightGBM",
    "LSTM", "PatchTST",
    "Informer(自分)", "Informer(論文)",
]

print()
print("=" * 70)
print("4. 正規化スケールでの全モデル比較")
print("   （論文と同じスケール）")
print("   論文Informer: 24h MAE=0.247 MSE=0.098, 168h MAE=0.346 MSE=0.183")
print("=" * 70)
print()
print(f"{'Model':22s}  {'24h MAE':>8s}  {'24h MSE':>8s}  {'168h MAE':>9s}  {'168h MSE':>9s}")
print("-" * 72)

for model in all_models:
    if model == "Informer(論文)":
        r24, r168 = paper[24], paper[168]
    else:
        r24, r168 = norm_results[model][24], norm_results[model][168]
    print(f"{model:22s}  {r24['MAE']:8.4f}  {r24['MSE']:8.4f}  {r168['MAE']:9.4f}  {r168['MSE']:9.4f}")

print()
print("=" * 70)
print("参考: 元スケール（℃）での全モデル比較")
print("=" * 70)
print()
print(f"{'Model':22s}  {'24h MAE':>8s}  {'24h MSE':>8s}  {'168h MAE':>9s}  {'168h MSE':>9s}")
print("-" * 72)

for model in all_models:
    if model == "Informer(論文)":
        r24 = {"MAE": paper[24]["MAE"] * inf_std, "MSE": paper[24]["MSE"] * inf_std ** 2}
        r168 = {"MAE": paper[168]["MAE"] * inf_std, "MSE": paper[168]["MSE"] * inf_std ** 2}
    elif model == "Informer(自分)":
        r24, r168 = informer_orig[24], informer_orig[168]
    else:
        r24 = {"MAE": step_results[model][24]["MAE"], "MSE": step_results[model][24]["RMSE"] ** 2}
        r168 = {"MAE": step_results[model][168]["MAE"], "MSE": step_results[model][168]["RMSE"] ** 2}
    print(f"{model:22s}  {r24['MAE']:8.4f}  {r24['MSE']:8.4f}  {r168['MAE']:9.4f}  {r168['MSE']:9.4f}")

# ============================================================
# 5. 可視化
# ============================================================
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

plot_models = [
    "Persistence", "SeasonalNaive24",
    "Ridge", "LightGBM",
    "LSTM", "PatchTST",
    "Informer(自分)",
]
colors = {
    "Persistence": "gray", "SeasonalNaive24": "silver",
    "Ridge": "tab:purple", "LightGBM": "tab:green",
    "LSTM": "tab:red", "PatchTST": "tab:blue",
    "Informer(自分)": "tab:orange",
}

horizons_plot = [24, 168]
x = np.arange(len(horizons_plot))
width = 0.11

fig, ax = plt.subplots(figsize=(14, 7))
for i, model in enumerate(plot_models):
    vals = [norm_results[model][h]["MAE"] for h in horizons_plot]
    ax.bar(x + i * width, vals, width, label=model, color=colors[model], alpha=0.85)

# 論文Informerの値を点線で追加
for j, h in enumerate(horizons_plot):
    paper_mae = paper[h]["MAE"]
    ax.hlines(paper_mae, x[j] - 0.1, x[j] + len(plot_models) * width + 0.05,
              color="tab:orange", linewidth=2, linestyle="--", alpha=0.6)
    ax.annotate(
        f"Paper Informer {h}h: {paper_mae:.3f}",
        xy=(x[j] + width * (len(plot_models) - 1) / 2, paper_mae),
        xytext=(0, 10), textcoords="offset points",
        fontsize=10, ha="center", color="tab:orange", fontweight="bold",
    )

ax.set_title("MAE Comparison — Normalized Scale (same scale as paper)")
ax.set_xlabel("Prediction Horizon")
ax.set_ylabel("MAE (normalized scale)")
ax.set_xticks(x + width * (len(plot_models) - 1) / 2)
ax.set_xticklabels([f"{h}h" for h in horizons_plot])
ax.legend(fontsize=10, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig("outputs/figures/normalized_comparison.png")
plt.close()
print(f"\nSaved: outputs/figures/normalized_comparison.png")

print("\n>>> 完了")
