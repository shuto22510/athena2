"""Step 2 追加可視化: 特徴量重要度比較 + 予測vs実測プロット"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

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

# --- データ読み込み・分割 ---
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df.set_index("date", inplace=True)

TRAIN_END = 12 * 30 * 24
VAL_END = TRAIN_END + 4 * 30 * 24


# --- 特徴量生成 ---
def build_features(df):
    feat = pd.DataFrame(index=df.index)
    for lag in [1, 2, 3, 6, 12, 24, 168]:
        feat[f"OT_lag{lag}"] = df["OT"].shift(lag)
    feat["OT_rolling24_mean"] = df["OT"].shift(1).rolling(24).mean()
    hour = df.index.hour
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    month = df.index.month
    feat["month_sin"] = np.sin(2 * np.pi * month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * month / 12)
    feat["HULL"] = df["HULL"]
    feat["MULL"] = df["MULL"]
    return feat


features = build_features(df)
feature_names = features.columns.tolist()

# --- 学習（horizon=1h, 24h でLightGBM + Ridge） ---
models_by_horizon = {}

for h in [1, 24]:
    target = df["OT"].shift(-h)
    valid = features.notna().all(axis=1) & target.notna()
    X = features[valid]
    y = target[valid].values
    idx = features[valid].index

    train_mask = idx < df.index[TRAIN_END]
    test_mask = idx >= df.index[VAL_END]

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_idx = idx[test_mask]

    lgb = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=31, verbose=-1, n_jobs=1,
    )
    lgb.fit(X_train, y_train)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    models_by_horizon[h] = {
        "LightGBM": lgb,
        "Ridge": ridge,
        "X_test": X_test,
        "y_test": y_test,
        "test_idx": test_idx,
    }

# --- 可視化1: 特徴量重要度比較（LightGBM 1h vs 24h） ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, h in zip(axes, [1, 24]):
    lgb = models_by_horizon[h]["LightGBM"]
    importances = lgb.feature_importances_
    sorted_idx = np.argsort(importances)
    ax.barh(
        np.array(feature_names)[sorted_idx],
        importances[sorted_idx],
        color="tab:green" if h == 1 else "tab:blue",
        alpha=0.8,
    )
    ax.set_title(f"Horizon={h}h")
    ax.set_xlabel("Importance (split count)")

fig.suptitle("Feature Importance Comparison — LightGBM", fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR + "feature_importance_comparison.png")
plt.close()
print(f"Saved: {FIGURES_DIR}feature_importance_comparison.png")

# --- 可視化2: 予測vs実測（1h, 24hの2パネル） ---
ot = df["OT"].values
n_plot = 14 * 24

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

for ax, h in zip(axes, [1, 24]):
    data = models_by_horizon[h]
    test_idx = data["test_idx"]
    y_test = data["y_test"]
    ridge_pred = data["Ridge"].predict(data["X_test"])

    # Persistence: OT[t]
    test_start_pos = np.searchsorted(df.index, test_idx[0])
    persist_pred = ot[test_start_pos : test_start_pos + len(y_test)]

    n = min(n_plot, len(y_test))
    ax.plot(test_idx[:n], y_test[:n], label="Actual", color="black", linewidth=1.2)
    ax.plot(test_idx[:n], persist_pred[:n], label="Persistence", color="gray",
            linewidth=0.8, linestyle="--", alpha=0.7)
    ax.plot(test_idx[:n], ridge_pred[:n], label="Ridge", color="tab:purple",
            linewidth=0.9, alpha=0.9)
    ax.set_title(f"Horizon={h}h (First 2 Weeks)")
    ax.set_ylabel("OT")
    ax.legend()

axes[1].set_xlabel("Date")
fig.suptitle("Prediction vs Actual — Persistence & Ridge", fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR + "prediction_vs_actual.png")
plt.close()
print(f"Saved: {FIGURES_DIR}prediction_vs_actual.png")
