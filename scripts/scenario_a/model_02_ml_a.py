"""Step 2: MLモデル (Scenario A: lag0利用可能)
- 特徴量に OT_lag0 (= OT(t)) を追加
- Ridge alpha の簡易探索を実施
"""
import os, sys
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

DATA_PATH = "data/ETTh1.csv"
HORIZONS = [1, 6, 24, 168]
TRAIN_END = 8640
VAL_END = 11520

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df.set_index("date", inplace=True)

def build_features(df):
    feat = pd.DataFrame(index=df.index)
    # lag 0 (current OT) を追加
    for lag in [0, 1, 2, 3, 6, 12, 24, 168]:
        feat[f"OT_lag{lag}"] = df["OT"].shift(lag)
    
    # t時点までの移動平均
    feat["OT_rolling24_mean"] = df["OT"].rolling(24).mean()
    
    # 時間・負荷
    feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    feat["HULL"] = df["HULL"]
    feat["MULL"] = df["MULL"]
    return feat

features = build_features(df)
results = []

for h in HORIZONS:
    print(f"Horizon: {h}h evaluation...")
    target = df["OT"].shift(-h)
    valid = features.notna().all(axis=1) & target.notna()
    
    X = features[valid]
    y = target[valid]
    
    X_train = X[X.index < df.index[TRAIN_END]]
    y_train = y[y.index < df.index[TRAIN_END]]
    X_test = X[X.index >= df.index[VAL_END]]
    y_test = y[y.index >= df.index[VAL_END]]

    # Ridge Tuning (Scenario A では lag0 が強いため alpha が小さい方が良い可能性がある)
    best_ridge_mae = float('inf')
    best_alpha = 1.0
    for alpha in [1.0, 100.0, 2000.0]:
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        pred = model.predict(X_test)
        m = np.mean(np.abs(y_test - pred))
        if m < best_ridge_mae:
            best_ridge_mae = m
            best_alpha = alpha
    
    results.append({"Model": f"Ridge(a={best_alpha})", "Horizon": f"{h}h", "MAE": best_ridge_mae})
    
    # LightGBM
    lgb = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, verbose=-1, n_jobs=1)
    lgb.fit(X_train, y_train)
    lgb_pred = lgb.predict(X_test)
    results.append({"Model": "LightGBM", "Horizon": f"{h}h", "MAE": np.mean(np.abs(y_test - lgb_pred))})

print("\n=== Scenario A ML Results ===")
print(pd.DataFrame(results).to_string(index=False))
