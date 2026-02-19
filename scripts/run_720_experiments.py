"""seq_len=720 再実験（軽量設定）
d_model=256, d_ff=1024, batch_size=16, epochs=4
結果は results_720/ に保存し、最後にMAE比較テーブルを出力
"""
import os, sys, json, time
import numpy as np
import pandas as pd

# ============================================================
# Part 1: Informer (subprocess で main_informer.py を呼ぶ)
# ============================================================
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
PYTHON = os.path.join(PROJECT, ".venv", "bin", "python")
INFORMER_DIR = os.path.join(PROJECT, "informer")
RESULTS_720 = os.path.join(PROJECT, "results_720")
os.makedirs(RESULTS_720, exist_ok=True)

# Log file for progress tracking
LOG = os.path.join(RESULTS_720, "experiment_log.txt")

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("=== seq_len=720 実験開始 ===")

informer_configs = [
    {"pred_len": 24,  "label_len": 168, "name": "informer_24"},
    {"pred_len": 168, "label_len": 336, "name": "informer_168"},
]

informer_results = {}

for cfg in informer_configs:
    log(f"Informer pred_len={cfg['pred_len']} 開始（推定: ~5分）")
    start = time.time()

    cmd = [
        PYTHON, "-u", "main_informer.py",
        "--model", "informer",
        "--data", "ETTh1",
        "--features", "S",
        "--seq_len", "720",
        "--label_len", str(cfg["label_len"]),
        "--pred_len", str(cfg["pred_len"]),
        "--enc_in", "1", "--dec_in", "1", "--c_out", "1",
        "--d_model", "256", "--n_heads", "8",
        "--e_layers", "2", "--d_layers", "1", "--d_ff", "1024",
        "--attn", "prob", "--factor", "5",
        "--embed", "timeF", "--distil",
        "--dropout", "0.05",
        "--itr", "1",
        "--train_epochs", "4",
        "--batch_size", "16",
        "--patience", "3",
        "--learning_rate", "0.0001",
        "--des", "poc_720",
        "--inverse",
        "--root_path", os.path.join(PROJECT, "data") + "/",
        "--checkpoints", os.path.join(PROJECT, "checkpoints") + "/",
    ]

    proc = subprocess.Popen(
        cmd, cwd=INFORMER_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "0"},
    )

    # Stream output line by line
    output_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        output_lines.append(line)
        # Print key lines
        if any(k in line for k in ["Epoch:", "mse:", "test shape", "speed:", "iters: 100", "iters: 200", "iters: 300", "iters: 400", "iters: 500"]):
            log(f"  {line}")

    proc.wait()
    elapsed = time.time() - start
    log(f"  完了 (exit={proc.returncode}, {elapsed:.0f}秒)")

    # Find and copy result files
    # Results are saved in informer/results/<setting_name>/
    informer_results_dir = os.path.join(INFORMER_DIR, "results")
    if os.path.exists(informer_results_dir):
        for d in os.listdir(informer_results_dir):
            if f"sl720" in d and f"pl{cfg['pred_len']}_" in d:
                src = os.path.join(informer_results_dir, d)
                pred_f = os.path.join(src, "pred.npy")
                true_f = os.path.join(src, "true.npy")
                if os.path.exists(pred_f):
                    pred = np.load(pred_f)
                    true = np.load(true_f)
                    mae = float(np.mean(np.abs(true[:, -1, 0] - pred[:, -1, 0])))
                    mse = float(np.mean((true[:, -1, 0] - pred[:, -1, 0]) ** 2))
                    informer_results[cfg["pred_len"]] = {"MAE": mae, "MSE": mse}
                    log(f"  → MAE={mae:.4f}, MSE={mse:.4f}")
                    # Copy to results_720
                    import shutil
                    dst = os.path.join(RESULTS_720, cfg["name"])
                    os.makedirs(dst, exist_ok=True)
                    shutil.copy2(pred_f, dst)
                    shutil.copy2(true_f, dst)
                break

    # Save progress
    with open(os.path.join(RESULTS_720, "informer_results.json"), "w") as f:
        json.dump({str(k): v for k, v in informer_results.items()}, f, indent=2)

log("")

# ============================================================
# Part 2: LSTM (lookback=720) — inline
# ============================================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"LSTM Device: {device}")

LOOKBACK = 720
HORIZONS = [1, 24, 168]
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10

df = pd.read_csv(os.path.join(PROJECT, "data", "ETTh1.csv"), parse_dates=["date"])
df.set_index("date", inplace=True)

TRAIN_END = 8640
VAL_END = 11520

feature_cols = ["OT", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
data = df[feature_cols].values.astype(np.float32)

train_data = data[:TRAIN_END]
mean_ = train_data.mean(axis=0)
std_ = train_data.std(axis=0)
data_norm = (data - mean_) / std_
ot_mean, ot_std = mean_[0], std_[0]


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


class LSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout, horizon):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_lstm(model, train_loader, val_loader, horizon_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        tloss, nb = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tloss += loss.item(); nb += 1

        model.eval()
        vloss, nv = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vloss += criterion(model(x), y).item(); nv += 1
        vloss /= nv
        scheduler.step(vloss)

        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            log(f"  Epoch {epoch+1:3d}  train={tloss/nb:.6f}  val={vloss:.6f}")

        if no_improve >= PATIENCE:
            log(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.to(device)


def evaluate_lstm(model, test_loader):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.append(model(x.to(device)).cpu().numpy())
            actuals.append(y.numpy())
    p = np.concatenate(preds) * ot_std + ot_mean
    a = np.concatenate(actuals) * ot_std + ot_mean
    return p, a


lstm_results = {}
for h in HORIZONS:
    log(f"LSTM horizon={h}h 開始（推定: ~5-7分）")
    start = time.time()

    train_ds = TimeSeriesDataset(data_norm, 0, TRAIN_END, LOOKBACK, h)
    val_ds = TimeSeriesDataset(data_norm, TRAIN_END, VAL_END, LOOKBACK, h)
    test_ds = TimeSeriesDataset(data_norm, VAL_END, len(data_norm), LOOKBACK, h)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    log(f"  Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    model = LSTMForecaster(len(feature_cols), HIDDEN_SIZE, NUM_LAYERS, DROPOUT, h).to(device)
    train_lstm(model, train_loader, val_loader, f"{h}h")

    preds, actuals = evaluate_lstm(model, test_loader)
    pred_h, actual_h = preds[:, -1], actuals[:, -1]
    mae = float(np.mean(np.abs(actual_h - pred_h)))
    rmse = float(np.sqrt(np.mean((actual_h - pred_h) ** 2)))
    lstm_results[h] = {"MAE": mae, "RMSE": rmse}
    elapsed = time.time() - start
    log(f"  → MAE={mae:.4f}, RMSE={rmse:.4f} ({elapsed:.0f}秒)")

with open(os.path.join(RESULTS_720, "lstm_results.json"), "w") as f:
    json.dump({str(k): v for k, v in lstm_results.items()}, f, indent=2)

# ============================================================
# Part 3: MAE比較テーブル
# ============================================================
log("")
log("=" * 70)
log("MAE比較テーブル（元スケール ℃）")
log("=" * 70)
log("")

# 旧結果
informer_old = {24: 2.6607, 168: 3.8778}
lstm_old = {1: 0.4726, 24: 2.7225, 168: 5.8927}

# 正規化用std
train_ot = data[:TRAIN_END, 0]
ot_std_ddof0 = float(train_ot.std())

header = f"{'Model':20s}  {'Horizon':>8s}  {'旧(168)':>10s}  {'新(720)':>10s}  {'改善':>8s}"
log(header)
log("-" * 65)

for h in [24, 168]:
    old = informer_old.get(h, float("nan"))
    new = informer_results.get(h, {}).get("MAE", float("nan"))
    diff = ((old - new) / old * 100) if not np.isnan(new) else float("nan")
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "")
    log(f"{'Informer':20s}  {str(h)+'h':>8s}  {old:10.4f}  {new:10.4f}  {diff:+7.1f}% {arrow}")

log("")
for h in [1, 24, 168]:
    old = lstm_old.get(h, float("nan"))
    new = lstm_results.get(h, {}).get("MAE", float("nan"))
    diff = ((old - new) / old * 100) if not np.isnan(new) else float("nan")
    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "")
    log(f"{'LSTM':20s}  {str(h)+'h':>8s}  {old:10.4f}  {new:10.4f}  {diff:+7.1f}% {arrow}")

# 正規化スケール
log("")
log("=" * 70)
log("正規化MAE比較（論文スケール）")
log("=" * 70)
log(f"  std(ddof=0) = {ot_std_ddof0:.6f}")
log("")

paper = {24: 0.247, 168: 0.346}
header2 = f"{'Model':20s}  {'Horizon':>8s}  {'旧(168)':>10s}  {'新(720)':>10s}  {'論文':>10s}"
log(header2)
log("-" * 65)

for h in [24, 168]:
    old_n = informer_old.get(h, 0) / ot_std_ddof0
    new_n = informer_results.get(h, {}).get("MAE", float("nan")) / ot_std_ddof0
    paper_n = paper.get(h, float("nan"))
    log(f"{'Informer':20s}  {str(h)+'h':>8s}  {old_n:10.4f}  {new_n:10.4f}  {paper_n:10.4f}")

log("")
for h in [1, 24, 168]:
    old_n = lstm_old.get(h, 0) / ot_std_ddof0
    new_n = lstm_results.get(h, {}).get("MAE", float("nan")) / ot_std_ddof0
    log(f"{'LSTM':20s}  {str(h)+'h':>8s}  {old_n:10.4f}  {new_n:10.4f}  {'---':>10s}")

log("")
log(">>> 全実験完了")
