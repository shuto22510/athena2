"""Step 2: MLモデル比較（LightGBM / XGBoost / RandomForest / Ridge）"""

# model_02_ml.py — ML評価（Ridge/LightGBM） Ridge α=2000 含む
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# --- 設定 ---
DATA_PATH = "data/ETTh1.csv"
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

# --- データ読み込み・分割 ---
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df.set_index("date", inplace=True)

TRAIN_END = 12 * 30 * 24  # 8640
VAL_END = TRAIN_END + 4 * 30 * 24  # 11520

print(f"Train: 0~{TRAIN_END - 1} ({TRAIN_END} samples)")
print(f"Val:   {TRAIN_END}~{VAL_END - 1} ({VAL_END - TRAIN_END} samples)")
print(f"Test:  {VAL_END}~{len(df) - 1} ({len(df) - VAL_END} samples)")
print()


# --- 特徴量生成 ---
def build_features(df):
    """EDA根拠に基づく特徴量を生成"""
    feat = pd.DataFrame(index=df.index)

    # OTラグ特徴量
    for lag in [1, 2, 3, 6, 12, 24, 168]:
        feat[f"OT_lag{lag}"] = df["OT"].shift(lag)

    # OT 24h移動平均
    feat["OT_rolling24_mean"] = df["OT"].shift(1).rolling(24).mean()

    # 時間特徴量（sin/cos）
    hour = df.index.hour
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # 月特徴量（sin/cos）
    month = df.index.month
    feat["month_sin"] = np.sin(2 * np.pi * month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * month / 12)

    # 負荷変数（OTとの相関が高い2つのみ）
    feat["HULL"] = df["HULL"]
    feat["MULL"] = df["MULL"]

    return feat


features = build_features(df)
feature_names = features.columns.tolist()
print(f"Features ({len(feature_names)}): {feature_names}")
print()


# --- 評価関数 ---
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# --- モデル定義 ---
def get_models():
    return {
        "LightGBM": LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=31, verbose=-1, n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            verbosity=0, n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=10, n_jobs=-1, random_state=42,
        ),
        "Ridge": Ridge(alpha=1.0),
    }


# --- 学習・予測・評価 ---
results = []
predictions = {}  # (model_name, horizon) -> pred array
trained_models = {}  # (model_name, horizon) -> fitted model

ot = df["OT"].values

for h in HORIZONS:
    print(f"=== Horizon: {h}h ===")

    # ターゲット: OT[t+h] を時刻tの特徴量で予測
    target = df["OT"].shift(-h)

    # 有効な行（特徴量にNaNがなく、ターゲットもある）
    valid = features.notna().all(axis=1) & target.notna()

    X = features[valid]
    y = target[valid].values
    idx = features[valid].index

    # 分割（インデックスベース）
    train_mask = idx < df.index[TRAIN_END]
    test_mask = idx >= df.index[VAL_END]

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_idx = idx[test_mask]

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # ベースライン（テストセット上で再計算）
    # Persistence: OT[t] = OT_lag0 ≒ テスト時のOT
    test_start_pos = np.searchsorted(df.index, test_idx[0])
    actual_test = y_test

    persist_pred = ot[test_start_pos : test_start_pos + len(y_test)]
    # SeasonalNaive logic (no logic leakage): yhat[t+h] = y[t - 24 + ((h-1)%24 + 1)]
    k = ((h - 1) % 24) + 1
    seasonal_pred = ot[test_start_pos - 24 + k : test_start_pos - 24 + k + len(y_test)]

    for name, pred in [("Persistence", persist_pred), ("SeasonalNaive24", seasonal_pred)]:
        n = min(len(pred), len(actual_test))
        results.append({
            "Model": name, "Horizon": f"{h}h",
            "MAE": mae(actual_test[:n], pred[:n]),
            "RMSE": rmse(actual_test[:n], pred[:n]),
        })

    # MLモデル
    models = get_models()
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        m = mae(y_test, pred)
        r = rmse(y_test, pred)
        results.append({
            "Model": name, "Horizon": f"{h}h", "MAE": m, "RMSE": r,
        })
        predictions[(name, h)] = (test_idx, pred, y_test)
        trained_models[(name, h)] = model
        print(f"  {name:15s}  MAE={m:.4f}  RMSE={r:.4f}")

    print()

# --- 結果テーブル ---
results_df = pd.DataFrame(results)
print("=== Full Results (Test Set) ===")
for h in HORIZONS:
    print(f"\n--- Horizon: {h}h ---")
    sub = results_df[results_df["Horizon"] == f"{h}h"].sort_values("MAE")
    for _, row in sub.iterrows():
        print(f"  {row['Model']:20s}  MAE={row['MAE']:.4f}  RMSE={row['RMSE']:.4f}")

# --- 可視化1: ホライズン別MAE棒グラフ ---
all_models = ["Persistence", "SeasonalNaive24", "LightGBM", "XGBoost", "RandomForest", "Ridge"]
colors = ["gray", "silver", "tab:green", "tab:blue", "tab:orange", "tab:purple"]
x = np.arange(len(HORIZONS))
width = 0.13

fig, ax = plt.subplots(figsize=(12, 6))
for i, (model, color) in enumerate(zip(all_models, colors)):
    vals = []
    for h in HORIZONS:
        sub = results_df[(results_df["Model"] == model) & (results_df["Horizon"] == f"{h}h")]
        vals.append(sub["MAE"].values[0] if len(sub) > 0 else 0)
    ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
ax.set_title("MAE Comparison: Baselines vs ML Models")
ax.set_xlabel("Horizon")
ax.set_ylabel("MAE")
ax.set_xticks(x + width * (len(all_models) - 1) / 2)
ax.set_xticklabels([f"{h}h" for h in HORIZONS])
ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(FIGURES_DIR + "ml_metrics.png")
plt.close()
print(f"\nSaved: {FIGURES_DIR}ml_metrics.png")

# --- 可視化2: 最良MLモデルの予測 vs 実績（horizon=24h, 最初2週間） ---
h_plot = 24
best_model_name = (
    results_df[
        (results_df["Horizon"] == f"{h_plot}h")
        & (~results_df["Model"].isin(["Persistence", "SeasonalNaive24"]))
    ]
    .sort_values("MAE")
    .iloc[0]["Model"]
)

test_idx_24, pred_24, actual_24 = predictions[(best_model_name, h_plot)]
n_plot = 14 * 24
plot_slice = slice(0, min(n_plot, len(actual_24)))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_idx_24[plot_slice], actual_24[plot_slice], label="Actual", color="black", linewidth=1.2)
ax.plot(test_idx_24[plot_slice], pred_24[plot_slice], label=best_model_name, color="tab:green", linewidth=0.9, alpha=0.9)
ax.set_title(f"{best_model_name} Predictions vs Actual (Horizon={h_plot}h, First 2 Weeks)")
ax.set_xlabel("Date")
ax.set_ylabel("OT")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR + "ml_predictions_24h.png")
plt.close()
print(f"Saved: {FIGURES_DIR}ml_predictions_24h.png")

# --- 可視化3: 特徴量重要度（LightGBM, horizon=24h） ---
lgb_model = trained_models[("LightGBM", 24)]
importances = lgb_model.feature_importances_
sorted_idx = np.argsort(importances)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    np.array(feature_names)[sorted_idx],
    importances[sorted_idx],
    color="tab:green",
    alpha=0.8,
)
ax.set_title("Feature Importance — LightGBM (Horizon=24h)")
ax.set_xlabel("Importance (split count)")
fig.tight_layout()
fig.savefig(FIGURES_DIR + "feature_importance_24h.png")
plt.close()
print(f"Saved: {FIGURES_DIR}feature_importance_24h.png")
