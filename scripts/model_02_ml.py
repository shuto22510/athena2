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

    # lag features (Scenario A: lag0 = OT(t) is available)
    for lag in [0, 1, 2, 3, 6, 12, 24, 168]:
        feat[f"OT_lag{lag}"] = df["OT"].shift(lag)

    # rolling mean up to time t (includes OT(t))
    feat["OT_rolling24_mean"] = df["OT"].rolling(24).mean()

    # time + load features at time t
    feat["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    feat["HULL"] = df["HULL"]
    feat["MULL"] = df["MULL"]
    return feat

features = build_features(df)
results = []

train_cut = df.index[TRAIN_END]
val_cut = df.index[VAL_END]

for h in HORIZONS:
    print(f"Horizon: {h}h evaluation...")

    target = df["OT"].shift(-h)  # y(t)=OT(t+h)
    valid = features.notna().all(axis=1) & target.notna()

    X = features[valid]
    y = target[valid]

    X_train = X[X.index < train_cut]
    y_train = y[y.index < train_cut]

    X_val = X[(X.index >= train_cut) & (X.index < val_cut)]
    y_val = y[(y.index >= train_cut) & (y.index < val_cut)]

    X_test = X[X.index >= val_cut]
    y_test = y[y.index >= val_cut]

    # --- Ridge tuning on VAL (NOT TEST) ---
    best_alpha, best_val_mae = None, float("inf")
    for alpha in [1.0, 100.0, 2000.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        val_mae = np.mean(np.abs(y_val - pred_val))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_alpha = alpha

    # Refit on Train+Val, evaluate on Test
    X_trainval = X[X.index < val_cut]
    y_trainval = y[y.index < val_cut]
    ridge = Ridge(alpha=best_alpha).fit(X_trainval, y_trainval)
    pred_test = ridge.predict(X_test)
    test_mae = np.mean(np.abs(y_test - pred_test))

    results.append({
        "Model": f"Ridge(a={best_alpha})",
        "Horizon": f"{h}h",
        "Val_MAE": best_val_mae,
        "Test_MAE": test_mae
    })

    # --- LightGBM (fixed params; no tuning in this PoC) ---
    lgb = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        verbose=-1, n_jobs=1, random_state=42
    )
    lgb.fit(X_trainval, y_trainval)
    lgb_pred = lgb.predict(X_test)
    results.append({
        "Model": "LightGBM",
        "Horizon": f"{h}h",
        "Val_MAE": np.nan,
        "Test_MAE": np.mean(np.abs(y_test - lgb_pred))
    })

print("\n=== Scenario A ML Results ===")
print(pd.DataFrame(results).to_string(index=False))
