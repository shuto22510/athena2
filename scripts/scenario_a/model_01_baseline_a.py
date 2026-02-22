"""Step 1: ベースラインモデル (Scenario A: lag0利用可能)
- ŷ(t+h) = y(t) (Persistence)
- ŷ(t+h) = y(t + h - 24*ceil(h/24)) (SeasonalNaive24)
"""
import os, sys
import pandas as pd
import numpy as np
import math

DATA_PATH = "data/ETTh1.csv"
HORIZONS = [1, 6, 24, 168]
TRAIN_END = 8640
VAL_END = 11520

# データ読み込み
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df.set_index("date", inplace=True)
ot = df["OT"].values

# 学習期間の平均
train_mean = ot[:TRAIN_END].mean()

results = []

for h in HORIZONS:
    # 予測対象 y[t+h]
    # t の範囲: VAL_END .. len(ot)-h-1
    n_test = len(ot) - VAL_END - h
    actual = ot[VAL_END + h : VAL_END + h + n_test]

    # Persistence: ŷ(t+h) = y(t)
    pred_persist = ot[VAL_END : VAL_END + n_test]

    # SeasonalNaive24: ŷ(t+h) = y(t + h - 24*ceil(h/24))
    k = math.ceil(h / 24)
    pred_seasonal = ot[VAL_END + h - 24*k : VAL_END + h - 24*k + n_test]

    # Mean
    pred_mean = np.full(n_test, train_mean)

    for name, pred in [("Persistence", pred_persist), ("SeasonalNaive24", pred_seasonal), ("Mean", pred_mean)]:
        results.append({
            "Model": name, "Horizon": f"{h}h",
            "MAE": np.mean(np.abs(actual - pred)),
            "RMSE": np.sqrt(np.mean((actual - pred)**2))
        })

# 表示
print("=== Scenario A Baseline Results ===")
rdf = pd.DataFrame(results)
for h in HORIZONS:
    print(f"\nHorizon: {h}h")
    print(rdf[rdf["Horizon"] == f"{h}h"].to_string(index=False))
