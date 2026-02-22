"""Step 2: MLモデル比較 (Operational Scenario A: Lag 0 利用可能)
- 運用シナリオA: 時刻tのOTを観測した直後にt+hを予測 (lag0利用可能)
- Ridge Alpha: Validationセットでチューニングし、Train+Valで再学習
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

# --- 設定 ---
DATA_PATH = "data/ETTh1.csv"
FIGURES_DIR = "outputs/figures/"
HORIZONS = [1, 6, 24, 168]
TRAIN_END = 8640
VAL_END = 11520

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
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# --- データ読み込み ---
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df.set_index("date", inplace=True)

# --- 特徴量生成 ---
def build_features(df):
    """Scenario A: 時刻tの情報(OT_lag0等)のみを使用"""
    feat = pd.DataFrame(index=df.index)

    # Lag features: lag0 (current observation) is available
    for lag in [0, 1, 2, 3, 6, 12, 24, 168]:
        feat[f"OT_lag{lag}"] = df["OT"].shift(lag)

    # OT 24h rolling mean up to time t (includes OT(t))
    feat["OT_rolling24_mean"] = df["OT"].rolling(24).mean()

    # Time attributes at time t
    feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    feat["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # Load features at time t
    feat["HULL"] = df["HULL"]
    feat["MULL"] = df["MULL"]
    return feat

features = build_features(df)
results = []
train_cut = df.index[TRAIN_END]
val_cut = df.index[VAL_END]

# --- 評価ループ ---
for h in HORIZONS:
    print(f"=== Horizon: {h}h ===")
    target = df["OT"].shift(-h)
    valid = features.notna().all(axis=1) & target.notna()

    X = features[valid]
    y = target[valid]

    X_train = X[X.index < train_cut]
    y_train = y[y.index < train_cut]
    X_val = X[(X.index >= train_cut) & (X.index < val_cut)]
    y_val = y[(y.index >= train_cut) & (y.index < val_cut)]
    X_test = X[X.index >= val_cut]
    y_test = y[y.index >= val_cut]

    # 1. Ridge Tuning on Validation set
    best_alpha, best_val_mae = None, float("inf")
    for alpha in [1.0, 100.0, 2000.0]:
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        m = np.mean(np.abs(y_val - model.predict(X_val)))
        if m < best_val_mae:
            best_val_mae = m
            best_alpha = alpha

    # 2. Refit on Train+Val, Evaluate on Test
    X_trainval = X[X.index < val_cut]
    y_trainval = y[y.index < val_cut]
    ridge = Ridge(alpha=best_alpha).fit(X_trainval, y_trainval)
    ridge_test_mae = np.mean(np.abs(y_test - ridge.predict(X_test)))
    
    results.append({
        "Model": f"Ridge(a={best_alpha})", "Horizon": f"{h}h",
        "MAE": ridge_test_mae, "RMSE": np.sqrt(np.mean((y_test - ridge.predict(X_test))**2))
    })
    print(f"  Ridge(a={best_alpha}) MAE={ridge_test_mae:.4f}")

    # 3. LightGBM (Train+Valで学習)
    lgb = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        verbose=-1, n_jobs=1, random_state=42
    )
    lgb.fit(X_trainval, y_trainval)
    lgb_test_mae = np.mean(np.abs(y_test - lgb.predict(X_test)))
    results.append({
        "Model": "LightGBM", "Horizon": f"{h}h",
        "MAE": lgb_test_mae, "RMSE": np.sqrt(np.mean((y_test - lgb.predict(X_test))**2))
    })
    print(f"  LightGBM MAE={lgb_test_mae:.4f}")

# --- 結果出力 ---
results_df = pd.DataFrame(results)
print("\n=== Unified Scenario A Results ===")
print(results_df.to_string(index=False))
