"""Step 3: LSTM 時系列予測モデル (Scenario A: Lag 0 利用可能)
- 運用シナリオA: 時刻tの情報 [t-lookback+1, ..., t] を入力とし、t+h を予測
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 再現性
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = "data/ETTh1.csv"
LOOKBACK = 168
HORIZONS = [1, 6, 24, 168]
BATCH_SIZE = 64
TRAIN_END = 8640
VAL_END = 11520
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データ読み込み
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df.set_index("date", inplace=True)
feature_cols = ["OT", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
data = df[feature_cols].values.astype(np.float32)

# 正規化 (Trainセットの統計量を使用)
train_data = data[:TRAIN_END]
m, s = train_data.mean(axis=0), train_data.std(axis=0)
data_norm = (data - m) / s
ot_m, ot_s = m[0], s[0]

# --- Dataset ---
class ScenarioADataset(Dataset):
    def __init__(self, data, start, end, lookback, horizon):
        self.data = data
        self.lookback = lookback
        self.horizon = horizon
        self.start = max(start, lookback - 1)
        self.end = min(end, len(data) - horizon)

    def __len__(self): return self.end - self.start

    def __getitem__(self, idx):
        t = self.start + idx
        # Scenario A: 時刻tの情報までを入力とする
        x = self.data[t - self.lookback + 1 : t + 1]
        # 予測対象は [t+1, ..., t+horizon]。最後のステップ(t+horizon)のみを評価対象とする
        y = self.data[t + 1 : t + 1 + self.horizon, 0]
        return torch.tensor(x), torch.tensor(y)

# --- Model ---
class LSTMForecaster(nn.Module):
    def __init__(self, dim, h_dim, out_dim):
        super().__init__()
        self.lstm = nn.LSTM(dim, h_dim, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 評価 ---
results = []
for h in HORIZONS:
    print(f"=== Horizon: {h}h ===")
    train_ds = ScenarioADataset(data_norm, 0, TRAIN_END, LOOKBACK, h)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = ScenarioADataset(data_norm, TRAIN_END, VAL_END, LOOKBACK, h)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_ds = ScenarioADataset(data_norm, VAL_END, len(data_norm), LOOKBACK, h)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    model = LSTMForecaster(len(feature_cols), 128, h).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    
    # 簡易評価のためエポック数を制限
    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    
    model.eval()
    errors = []
    with torch.no_grad():
        for x, y in test_loader:
            p = model(x.to(device)).cpu().numpy() * ot_s + ot_m
            a = y.numpy() * ot_s + ot_m
            errors.append(np.abs(a[:, -1] - p[:, -1]))
    
    mae = np.mean(np.concatenate(errors))
    print(f"  MAE={mae:.4f}")
    results.append({"Model": "LSTM", "Horizon": f"{h}h", "MAE": mae})

print("\n=== Scenario A LSTM Results ===")
print(pd.DataFrame(results).to_string(index=False))
