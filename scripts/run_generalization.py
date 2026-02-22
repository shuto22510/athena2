"""ETTm1/ETTm2 汎化性検証: SeasonalNaive vs Informer(seq=720)
15分粒度: 24h=96steps, 168h=672steps
ETTh1/ETTh2の結果と並べて全データセット比較テーブルを出力
"""
# run_generalization.py — ETTh2/m1/m2の汎化性検証
import os, sys, json, time, subprocess
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(ROOT)
PYTHON = os.path.join(PROJECT, ".venv", "bin", "python")
INFORMER_DIR = os.path.join(PROJECT, "informer")
RESULTS_DIR = os.path.join(PROJECT, "results_ettm")
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG = os.path.join(RESULTS_DIR, "experiment_log.txt")

# Clear old log
if os.path.exists(LOG):
    os.remove(LOG)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run_seasonal_naive(dataset_name, csv_path, season_steps, horizons_steps):
    """Run SeasonalNaive for a given dataset."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    ot = df["OT"].values.astype(np.float32)

    # Split: same as Informer paper for minute-level data
    if "ETTm" in dataset_name:
        TRAIN_END = 12 * 30 * 24 * 4   # 34560
        VAL_END = TRAIN_END + 4 * 30 * 24 * 4  # 46080
    else:
        TRAIN_END = 12 * 30 * 24   # 8640
        VAL_END = TRAIN_END + 4 * 30 * 24  # 11520

    train_ot = ot[:TRAIN_END]
    ot_mean = float(train_ot.mean())
    ot_std = float(train_ot.std())  # ddof=0

    log(f"{dataset_name}: {len(df)} rows, OT mean={ot_mean:.4f}, std={ot_std:.4f}")
    log(f"  Train: {TRAIN_END}, Val: {VAL_END-TRAIN_END}, Test: {len(df)-VAL_END}")

    results = {}
    PERIOD = 96  # 15min granularity: 24h = 96 steps
    for h_steps, h_label in horizons_steps:
        # Evaluate starting from VAL_END. Target is ot[t+h_steps]
        n_test = len(ot) - VAL_END - h_steps
        if n_test <= 0: continue
        
        actuals = ot[VAL_END + h_steps : VAL_END + h_steps + n_test]
        
        # SeasonalNaive logic: yhat[t+h] = y[t - 96 + ((h-1)%96 + 1)]
        k = ((h_steps - 1) % PERIOD) + 1
        start_pos = VAL_END - PERIOD + k
        preds = ot[start_pos : start_pos + n_test]

        mae_orig = float(np.mean(np.abs(actuals - preds)))
        mae_norm = mae_orig / ot_std
        results[h_label] = {"MAE_orig": mae_orig, "MAE_norm": mae_norm}
        log(f"  SeasonalNaive (horizon={h_label}): MAE={mae_orig:.4f} (norm={mae_norm:.4f})")

    return results, ot_std


def run_informer(dataset_name, data_key, pred_len, label_len, freq="h"):
    """Run Informer for one config via subprocess."""
    log(f"Informer pred_len={pred_len} 開始")
    start = time.time()

    # Use smaller batch for very long pred_len to avoid OOM on 4GB VRAM
    batch = 8 if pred_len >= 672 else 16

    cmd = [
        PYTHON, "-u", "main_informer.py",
        "--model", "informer",
        "--data", data_key,
        "--features", "S",
        "--seq_len", "720",
        "--label_len", str(label_len),
        "--pred_len", str(pred_len),
        "--enc_in", "1", "--dec_in", "1", "--c_out", "1",
        "--d_model", "256", "--n_heads", "8",
        "--e_layers", "2", "--d_layers", "1", "--d_ff", "1024",
        "--attn", "prob", "--factor", "5",
        "--embed", "timeF", "--distil",
        "--dropout", "0.05",
        "--freq", freq,
        "--itr", "1",
        "--train_epochs", "4",
        "--batch_size", str(batch),
        "--patience", "3",
        "--learning_rate", "0.0001",
        "--des", f"{data_key.lower()}_720",
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

    for line in proc.stdout:
        line = line.rstrip()
        if any(k in line for k in ["Epoch:", "mse:", "test shape",
                                     "iters: 100", "iters: 200", "iters: 300",
                                     "iters: 400", "iters: 500"]):
            log(f"  {line}")

    proc.wait()
    elapsed = time.time() - start
    log(f"  完了 (exit={proc.returncode}, {elapsed:.0f}秒)")

    return elapsed


def read_informer_result(data_key, pred_len, ot_std):
    """Read informer .npy result files."""
    informer_results_dir = os.path.join(INFORMER_DIR, "results")
    if not os.path.exists(informer_results_dir):
        return None
    for d in sorted(os.listdir(informer_results_dir)):
        if "sl720" in d and f"pl{pred_len}_" in d and f"{data_key.lower()}_720" in d:
            pred_f = os.path.join(informer_results_dir, d, "pred.npy")
            true_f = os.path.join(informer_results_dir, d, "true.npy")
            if os.path.exists(pred_f):
                pred = np.load(pred_f)
                true = np.load(true_f)
                mae_orig = float(np.mean(np.abs(true[:, -1, 0] - pred[:, -1, 0])))
                mae_norm = mae_orig / ot_std
                log(f"  → MAE={mae_orig:.4f} (norm={mae_norm:.4f})")
                return {"MAE_orig": mae_orig, "MAE_norm": mae_norm}
    return None


# ============================================================
# Run experiments
# ============================================================
all_results = {}

# --- ETTm1 ---
log("=" * 70)
log("ETTm1 (15分粒度, ETTh1と同じ変電所)")
log("=" * 70)

csv_m1 = os.path.join(PROJECT, "data", "ETTm1.csv")
sn_m1, std_m1 = run_seasonal_naive("ETTm1", csv_m1, 96,
                                     [(96, "24h"), (672, "168h")])

# Clean old informer results
import shutil
inf_res = os.path.join(INFORMER_DIR, "results")
if os.path.exists(inf_res):
    shutil.rmtree(inf_res)

inf_m1 = {}
for pl, ll in [(96, 48), (672, 336)]:
    run_informer("ETTm1", "ETTm1", pl, ll)
    r = read_informer_result("ETTm1", pl, std_m1)
    if r:
        h_label = "24h" if pl == 96 else "168h"
        inf_m1[h_label] = r

# Clean results before next dataset
if os.path.exists(inf_res):
    shutil.rmtree(inf_res)

all_results["ETTm1"] = {"seasonal": sn_m1, "informer": inf_m1, "std": std_m1}
log("")

# --- ETTm2 ---
log("=" * 70)
log("ETTm2 (15分粒度, 別の変電所)")
log("=" * 70)

csv_m2 = os.path.join(PROJECT, "data", "ETTm2.csv")
sn_m2, std_m2 = run_seasonal_naive("ETTm2", csv_m2, 96,
                                     [(96, "24h"), (672, "168h")])

inf_m2 = {}
for pl, ll in [(96, 48), (672, 336)]:
    run_informer("ETTm2", "ETTm2", pl, ll)
    r = read_informer_result("ETTm2", pl, std_m2)
    if r:
        h_label = "24h" if pl == 96 else "168h"
        inf_m2[h_label] = r

all_results["ETTm2"] = {"seasonal": sn_m2, "informer": inf_m2, "std": std_m2}

# ============================================================
# 全データセット比較テーブル
# ============================================================
log("")
log("=" * 80)
log("全データセット比較: SeasonalNaive vs Informer(720) — 正規化MAE")
log("=" * 80)
log("")

# ETTh1/ETTh2 results (from prior experiments)
prev = {
    "ETTh1": {
        "seasonal": {"24h": {"MAE_norm": 1.6324 / 9.1765}, "168h": {"MAE_norm": 2.7617 / 9.1765}},
        "informer": {"24h": {"MAE_norm": 1.9437 / 9.1765}, "168h": {"MAE_norm": 3.6599 / 9.1765}},
    },
    "ETTh2": {
        "seasonal": {"24h": {"MAE_norm": 0.2745}, "168h": {"MAE_norm": 0.4848}},
        "informer": {"24h": {"MAE_norm": 0.3735}, "168h": {"MAE_norm": 0.4099}},
    },
}

header = f"{'Dataset':10s}  {'Granularity':>12s}  {'Horizon':>8s}  {'SeasonalN':>10s}  {'Informer':>10s}  {'Winner':>15s}"
log(header)
log("-" * 80)

datasets = [
    ("ETTh1", "1h", prev["ETTh1"]),
    ("ETTh2", "1h", prev["ETTh2"]),
    ("ETTm1", "15min", all_results["ETTm1"]),
    ("ETTm2", "15min", all_results["ETTm2"]),
]

for ds_name, gran, ds in datasets:
    for h in ["24h", "168h"]:
        sn_v = ds["seasonal"].get(h, {}).get("MAE_norm", float("nan"))
        inf_v = ds["informer"].get(h, {}).get("MAE_norm", float("nan"))
        winner = "SeasonalNaive" if sn_v < inf_v else "Informer"
        log(f"{ds_name:10s}  {gran:>12s}  {h:>8s}  {sn_v:10.4f}  {inf_v:10.4f}  {winner:>15s}")
    log("")

# Save all results
with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
    # Serialize
    out = {}
    for k, v in {**prev, **all_results}.items():
        out[k] = {}
        for mk in ["seasonal", "informer"]:
            out[k][mk] = {}
            if mk in v:
                for h, vals in v[mk].items():
                    out[k][mk][h] = vals
    json.dump(out, f, indent=2)

log(">>> 全実験完了")
