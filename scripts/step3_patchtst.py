"""Step 3: PatchTST 時系列予測モデル（簡易自前実装）"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- 再現性 ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --- デバイス ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- 設定 ---
DATA_PATH = "data/ETTh1.csv"
FIGURES_DIR = "outputs/figures/"
LOOKBACK = 168
HORIZONS = [1, 6, 24, 168]
BATCH_SIZE = 64
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.1
PATCH_LEN = 24
STRIDE = 24
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10

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

# --- データ読み込み ---
df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df.set_index("date", inplace=True)

TRAIN_END = 12 * 30 * 24
VAL_END = TRAIN_END + 4 * 30 * 24

feature_cols = ["OT", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
data = df[feature_cols].values.astype(np.float32)

train_data = data[:TRAIN_END]
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
data_norm = (data - mean) / std

ot_mean, ot_std = mean[0], std[0]
n_features = len(feature_cols)

print(f"Data shape: {data.shape}, Features: {n_features}")
print(f"Patches per sample: {(LOOKBACK - PATCH_LEN) // STRIDE + 1}")
print()


# --- Dataset ---
class TimeSeriesDataset(Dataset):
    def __init__(self, data, start, end, lookback, horizon):
        self.data = data
        self.lookback = lookback
        self.horizon = horizon
        self.start = max(start, lookback)
        self.end = min(end, len(data) - horizon)

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        t = self.start + idx
        x = self.data[t - self.lookback : t]
        y = self.data[t : t + self.horizon, 0]
        return torch.tensor(x), torch.tensor(y)


# --- PatchTST Model ---
class PatchTST(nn.Module):
    def __init__(self, n_features, d_model, nhead, num_layers, dropout,
                 patch_len, stride, lookback, horizon):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (lookback - patch_len) // stride + 1

        # パッチ埋め込み: (patch_len * n_features) → d_model
        self.patch_embed = nn.Linear(patch_len * n_features, d_model)

        # 位置エンコーディング（学習可能）
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 予測ヘッド: 全パッチの出力をflattenして予測
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        # x: (batch, lookback, n_features)
        batch_size = x.shape[0]

        # パッチ化: (batch, n_patches, patch_len, n_features)
        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            patches.append(x[:, start:start + self.patch_len, :])
        patches = torch.stack(patches, dim=1)  # (B, n_patches, patch_len, n_feat)
        patches = patches.reshape(batch_size, self.n_patches, -1)  # (B, n_patches, patch_len*n_feat)

        # 埋め込み + 位置エンコーディング
        z = self.patch_embed(patches) + self.pos_embed

        # Transformer
        z = self.encoder(z)

        # 予測
        return self.head(z)


# --- 学習関数 ---
def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches

        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
                n_val += 1
        val_loss /= n_val

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            print(f"  Epoch {epoch+1:3d}  train={train_loss:.6f}  val={val_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.1e}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.to(device)
    return train_losses, val_losses


# --- 評価関数 ---
def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_actuals = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            all_preds.append(pred)
            all_actuals.append(y.numpy())
    preds = np.concatenate(all_preds, axis=0)
    actuals = np.concatenate(all_actuals, axis=0)
    preds = preds * ot_std + ot_mean
    actuals = actuals * ot_std + ot_mean
    return preds, actuals


# --- 全ホライズンで学習・評価 ---
results = []
all_train_losses = {}
all_val_losses = {}
predictions = {}

for h in HORIZONS:
    print(f"=== Horizon: {h}h ===")

    train_ds = TimeSeriesDataset(data_norm, 0, TRAIN_END, LOOKBACK, h)
    val_ds = TimeSeriesDataset(data_norm, TRAIN_END, VAL_END, LOOKBACK, h)
    test_ds = TimeSeriesDataset(data_norm, VAL_END, len(data_norm), LOOKBACK, h)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    model = PatchTST(
        n_features, D_MODEL, NHEAD, NUM_LAYERS, DROPOUT,
        PATCH_LEN, STRIDE, LOOKBACK, h,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    train_losses, val_losses = train_model(model, train_loader, val_loader)
    all_train_losses[h] = train_losses
    all_val_losses[h] = val_losses

    preds, actuals = evaluate(model, test_loader)
    pred_h = preds[:, -1]
    actual_h = actuals[:, -1]

    m = np.mean(np.abs(actual_h - pred_h))
    r = np.sqrt(np.mean((actual_h - pred_h) ** 2))
    results.append({"Model": "PatchTST", "Horizon": f"{h}h", "MAE": m, "RMSE": r})
    predictions[h] = (pred_h, actual_h)
    print(f"  MAE={m:.4f}  RMSE={r:.4f}")
    print()

# --- 結果テーブル ---
print("=== PatchTST Results (Test Set) ===")
for r in results:
    print(f"  {r['Horizon']:5s}  MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}")

# --- 可視化1: 学習曲線 ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, h in zip(axes.flatten(), HORIZONS):
    ax.plot(all_train_losses[h], label="Train", linewidth=0.8)
    ax.plot(all_val_losses[h], label="Val", linewidth=0.8)
    ax.set_title(f"Horizon={h}h")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
fig.suptitle("PatchTST Learning Curves", fontsize=16, y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR + "patchtst_learning_curve.png")
plt.close()
print(f"\nSaved: {FIGURES_DIR}patchtst_learning_curve.png")

# --- 可視化2: 予測vs実測（horizon=24h, 最初2週間） ---
n_plot = 14 * 24
pred_24, actual_24 = predictions[24]
n = min(n_plot, len(pred_24))
test_start_idx = VAL_END + LOOKBACK + 24 - 1
plot_index = df.index[test_start_idx : test_start_idx + n]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(plot_index, actual_24[:n], label="Actual", color="black", linewidth=1.2)
ax.plot(plot_index, pred_24[:n], label="PatchTST", color="tab:blue", linewidth=0.9, alpha=0.9)
ax.set_title("PatchTST Prediction vs Actual (Horizon=24h, First 2 Weeks)")
ax.set_xlabel("Date")
ax.set_ylabel("OT")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR + "patchtst_prediction_24h.png")
plt.close()
print(f"Saved: {FIGURES_DIR}patchtst_prediction_24h.png")

# 結果をファイルに保存
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/patchtst_results.csv", index=False)
print("Saved: outputs/patchtst_results.csv")
