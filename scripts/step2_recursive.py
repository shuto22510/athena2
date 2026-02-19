"""Step 2.5: Recursive Forecasting（1hモデルを再帰的に適用して長期予測）"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

# --- 設定 ---
DATA_PATH = "data/ETTh1.csv"
FIGURES_DIR = "outputs/figures/"
HORIZONS = [1, 6, 24, 168]
LAG_LIST = [1, 2, 3, 6, 12, 24, 168]
MAX_LAG = max(LAG_LIST)
# feature order: OT_lag1..168, OT_rolling24_mean, hour_sin, hour_cos, month_sin, month_cos, HULL, MULL
N_FEATURES = len(LAG_LIST) + 1 + 4 + 2  # 14

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
hull = df["HULL"].values
mull = df["MULL"].values
hours = df.index.hour.values
months = df.index.month.values

TRAIN_END = 12 * 30 * 24
VAL_END = TRAIN_END + 4 * 30 * 24

print(f"Train: 0~{TRAIN_END - 1}")
print(f"Val:   {TRAIN_END}~{VAL_END - 1}")
print(f"Test:  {VAL_END}~{len(df) - 1}")
print()


# --- 特徴量生成（学習用） ---
def build_features_array(ot, hull, mull, hours, months, n):
    """numpy配列で高速に特徴量生成"""
    feat = np.full((n, N_FEATURES), np.nan)
    for i, lag in enumerate(LAG_LIST):
        feat[lag:, i] = ot[:n - lag]
    # rolling24_mean (shift(1) then rolling(24))
    for t in range(25, n):
        feat[t, len(LAG_LIST)] = np.mean(ot[t - 25:t - 1])
    # hour sin/cos
    feat[:, len(LAG_LIST) + 1] = np.sin(2 * np.pi * hours[:n] / 24)
    feat[:, len(LAG_LIST) + 2] = np.cos(2 * np.pi * hours[:n] / 24)
    # month sin/cos
    feat[:, len(LAG_LIST) + 3] = np.sin(2 * np.pi * months[:n] / 12)
    feat[:, len(LAG_LIST) + 4] = np.cos(2 * np.pi * months[:n] / 12)
    # HULL, MULL
    feat[:, len(LAG_LIST) + 5] = hull[:n]
    feat[:, len(LAG_LIST) + 6] = mull[:n]
    return feat


n = len(df)
X_full = build_features_array(ot, hull, mull, hours, months, n)

# target = OT[t+1]
y_full = np.full(n, np.nan)
y_full[:n - 1] = ot[1:]

# 有効行
valid = ~np.isnan(X_full).any(axis=1) & ~np.isnan(y_full)
train_valid = valid.copy()
train_valid[TRAIN_END:] = False

X_train = X_full[train_valid]
y_train = y_full[train_valid]
print(f"Training samples: {len(X_train)}")

# --- 1hモデルを学習 ---
models = {
    "Ridge_Recursive": Ridge(alpha=1.0),
    "LightGBM_Recursive": LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=31, verbose=-1, n_jobs=1,
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"Trained: {name}")
print()


# --- Recursive予測（numpy最適化版） ---
def recursive_forecast_batch(model, ot, hull, mull, hours, months, start_pos, horizon):
    """start_posを起点にhorizonステップ先まで再帰予測（numpy版）"""
    buf = ot[max(0, start_pos - MAX_LAG - 24):start_pos].tolist()
    hull_val = hull[start_pos - 1]
    mull_val = mull[start_pos - 1]
    base_hour = hours[start_pos]
    base_month = months[start_pos]

    x = np.zeros(N_FEATURES)
    preds = []

    for step in range(horizon):
        cur_len = len(buf)

        # ラグ特徴量
        for i, lag in enumerate(LAG_LIST):
            x[i] = buf[cur_len - lag] if cur_len >= lag else 0.0

        # rolling24_mean
        if cur_len >= 25:
            x[len(LAG_LIST)] = np.mean(buf[cur_len - 25:cur_len - 1])
        elif cur_len >= 2:
            x[len(LAG_LIST)] = np.mean(buf[:cur_len - 1])
        else:
            x[len(LAG_LIST)] = 0.0

        # 時間特徴量
        h_idx = start_pos + step
        if h_idx < len(hours):
            cur_hour = hours[h_idx]
            cur_month = months[h_idx]
        else:
            cur_hour = (base_hour + step) % 24
            cur_month = base_month

        x[len(LAG_LIST) + 1] = np.sin(2 * np.pi * cur_hour / 24)
        x[len(LAG_LIST) + 2] = np.cos(2 * np.pi * cur_hour / 24)
        x[len(LAG_LIST) + 3] = np.sin(2 * np.pi * cur_month / 12)
        x[len(LAG_LIST) + 4] = np.cos(2 * np.pi * cur_month / 12)

        x[len(LAG_LIST) + 5] = hull_val
        x[len(LAG_LIST) + 6] = mull_val

        pred = model.predict(x.reshape(1, -1))[0]
        preds.append(pred)
        buf.append(pred)

    return preds


# --- 評価関数 ---
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# --- テストセットでRecursive予測 ---
results = []
recursive_preds = {}

test_start = VAL_END
test_end = len(df)

print("Running recursive forecasts...")

for model_name, model in models.items():
    print(f"\n  {model_name}:")
    for h in HORIZONS:
        t0 = time.time()
        n_test = test_end - test_start - h
        preds = np.zeros(n_test)
        actuals = ot[test_start + h:test_start + h + n_test]

        for i, t in enumerate(range(test_start, test_end - h)):
            forecast = recursive_forecast_batch(model, ot, hull, mull, hours, months, t, h)
            preds[i] = forecast[-1]

        m = mae(actuals, preds)
        r = rmse(actuals, preds)
        elapsed = time.time() - t0
        results.append({
            "Model": model_name, "Horizon": f"{h}h", "MAE": m, "RMSE": r,
        })
        test_indices = df.index[test_start + h:test_start + h + n_test]
        recursive_preds[(model_name, h)] = (test_indices, preds, actuals)
        print(f"    {h}h: MAE={m:.4f}  RMSE={r:.4f}  ({elapsed:.1f}s)")

# --- ベースラインを追加 ---
for h in HORIZONS:
    n_test = test_end - test_start - h
    actual = ot[test_start + h:test_start + h + n_test]
    persist_pred = ot[test_start:test_start + n_test]
    seasonal_pred = ot[test_start + h - 24:test_start + h - 24 + n_test]

    for name, pred in [("Persistence", persist_pred), ("SeasonalNaive24", seasonal_pred)]:
        results.append({
            "Model": name, "Horizon": f"{h}h",
            "MAE": mae(actual, pred), "RMSE": rmse(actual, pred),
        })

# --- 結果テーブル ---
results_df = pd.DataFrame(results)
print("\n=== Full Results: Recursive vs Baselines (Test Set) ===")
for h in HORIZONS:
    print(f"\n--- Horizon: {h}h ---")
    sub = results_df[results_df["Horizon"] == f"{h}h"].sort_values("MAE")
    for _, row in sub.iterrows():
        print(f"  {row['Model']:25s}  MAE={row['MAE']:.4f}  RMSE={row['RMSE']:.4f}")

# --- 可視化1: Direct vs Recursive MAE比較 ---
direct_results = {
    "Ridge_Direct": {"1h": 0.6673, "6h": 1.2674, "24h": 1.9343, "168h": 2.7513},
    "LightGBM_Direct": {"1h": 0.7905, "6h": 1.5614, "24h": 3.1435, "168h": 6.3294},
}

all_models_plot = [
    "Persistence", "SeasonalNaive24",
    "Ridge_Direct", "Ridge_Recursive",
    "LightGBM_Direct", "LightGBM_Recursive",
]
colors_plot = ["gray", "silver", "tab:purple", "tab:red", "tab:green", "tab:orange"]

x = np.arange(len(HORIZONS))
width = 0.13

fig, ax = plt.subplots(figsize=(14, 6))
for i, (model_plot, color) in enumerate(zip(all_models_plot, colors_plot)):
    vals = []
    for h in HORIZONS:
        hkey = f"{h}h"
        if model_plot in direct_results:
            vals.append(direct_results[model_plot][hkey])
        else:
            sub = results_df[(results_df["Model"] == model_plot) & (results_df["Horizon"] == hkey)]
            vals.append(sub["MAE"].values[0] if len(sub) > 0 else 0)
    ax.bar(x + i * width, vals, width, label=model_plot, color=color, alpha=0.85)

ax.set_title("MAE: Direct vs Recursive Forecasting")
ax.set_xlabel("Horizon")
ax.set_ylabel("MAE")
ax.set_xticks(x + width * (len(all_models_plot) - 1) / 2)
ax.set_xticklabels([f"{h}h" for h in HORIZONS])
ax.legend(fontsize=9, ncol=2)
fig.tight_layout()
fig.savefig(FIGURES_DIR + "direct_vs_recursive.png")
plt.close()
print(f"\nSaved: {FIGURES_DIR}direct_vs_recursive.png")

# --- 可視化2: Recursive予測 vs 実測（horizon=24h, 最初2週間） ---
n_plot = 14 * 24

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
for ax, model_name in zip(axes, ["Ridge_Recursive", "LightGBM_Recursive"]):
    indices, preds, actuals = recursive_preds[(model_name, 24)]
    n_p = min(n_plot, len(preds))
    ax.plot(indices[:n_p], actuals[:n_p], label="Actual", color="black", linewidth=1.2)
    ax.plot(indices[:n_p], preds[:n_p], label=model_name, linewidth=0.9, alpha=0.9)
    ax.set_title(f"{model_name} — Horizon=24h (First 2 Weeks)")
    ax.set_ylabel("OT")
    ax.legend()

axes[1].set_xlabel("Date")
fig.tight_layout()
fig.savefig(FIGURES_DIR + "recursive_prediction_24h.png")
plt.close()
print(f"Saved: {FIGURES_DIR}recursive_prediction_24h.png")
