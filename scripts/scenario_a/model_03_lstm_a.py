"""Step 3: LSTM (Scenario A: lag0利用可能)
- 窓関数の整合性を修正: 時刻tの情報を入力に含め、t+1..t+hを予測
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 簡略化して1ホライズン(24h)のみ検証、または全ホライズン
DATA_PATH = "data/ETTh1.csv"
HORIZONS = [1, 6, 24, 168]
LOOKBACK = 168
BATCH_SIZE = 64
TRAIN_END = 8640
VAL_END = 11520
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df.set_index("date", inplace=True)
data = df[["OT", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]].values.astype(np.float32)

# 正規化
train_data = data[:TRAIN_END]
m, s = train_data.mean(axis=0), train_data.std(axis=0)
data_norm = (data - m) / s
ot_m, ot_s = m[0], s[0]

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
        x = self.data[t - self.lookback + 1 : t + 1] # tまで含む
        y = self.data[t + 1 : t + 1 + self.horizon, 0] # t+1からhステップ
        return torch.tensor(x), torch.tensor(y)

class LSTM(nn.Module):
    def __init__(self, dim, h_dim, out_dim):
        super().__init__()
        self.lstm = nn.LSTM(dim, h_dim, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(h_dim, out_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

results = []
for h in HORIZONS:
    # 簡単のため 5 epoch のみのクイック評価
    train_ds = ScenarioADataset(data_norm, 0, TRAIN_END, LOOKBACK, h)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_ds = ScenarioADataset(data_norm, VAL_END, len(data_norm), LOOKBACK, h)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    model = LSTM(7, 128, h).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    
    for epoch in range(5):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    
    model.eval()
    all_mae = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            p = model(x).cpu().numpy() * ot_s + ot_m
            a = y.numpy() * ot_s + ot_m
            all_mae.append(np.abs(a[:, -1] - p[:, -1]))
    results.append({"Model": "LSTM", "Horizon": f"{h}h", "MAE": np.mean(np.concatenate(all_mae))})

print("\n=== Scenario A LSTM Results (Quick) ===")
print(pd.DataFrame(results).to_string(index=False))
