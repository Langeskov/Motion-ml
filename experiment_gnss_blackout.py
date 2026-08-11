"""
experiment_gnss_blackout.py
============================
A/B Test: NaN→0 vs NaN→Mask 在 GNSS 丢星场景下的对比。

实验管线 (不修改 data/raw/, 不修改主管线代码):
  1. 从已有 Session 中动态模拟 GNSS blackout (2s/5s/10s/30s)
  2. 训练两个模型:
     Baseline: 10D base → 21D derived (NaN→0)
     Mask:     11D base → 21D derived + gps_valid = 22D (NaN→0 + gps_valid)
  3. 对比 Accuracy / Macro F1 / Confusion Matrix / Per-duration 指标

输出:
  experiments/gnss_blackout/
  ├── results.csv
  └── run_*.json
"""

import json, os, sys, time
from dataclasses import dataclass, field

import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dataset.session_loader import SessionDataset, Session
from dataset.time_aligner import TimeAligner
from dataset.feature_builder import FeatureBuilder, FeatureSpec, DEFAULT_SPECS, FeatureMap
from dataset.preprocess import FeatureExtractor
from models.motion_lstm import MotionLSTM
from dataset.session_dataset import LABEL_MAP

# ── Config ──
@dataclass
class ExperimentConfig:
    data_dir: str = "data/raw"
    output_dir: str = "experiments/gnss_blackout"
    window_size: int = 150
    stride: int = 50
    hidden_size: int = 64
    num_layers: int = 1
    epochs: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    train_ratio: float = 0.8
    seed: int = 42
    blackout_durations_s: list[float] = field(default_factory=lambda: [2.0, 5.0, 10.0, 30.0])
    device: str = ""

# ── GNSS Blackout ──
def apply_gnss_blackout(gnss, imu, duration_s, seed=0):
    t0, t1 = imu["timestamp"].min(), imu["timestamp"].max()
    dur_ms = duration_s * 1000
    margin_ms = 5000
    rng = np.random.RandomState(seed)
    if t1 - t0 > dur_ms + 2 * margin_ms:
        start = float(rng.uniform(t0 + margin_ms, t1 - margin_ms - dur_ms))
    else:
        start = t0 + (t1 - t0 - dur_ms) / 2
    return gnss[~gnss["timestamp"].between(start, start + dur_ms)].copy()

def _sliding_windows(features, labels, window_size, stride):
    X, Y = [], []
    n = int(len(features))
    for s in range(0, n - window_size + 1, stride):
        e = s + window_size
        X.append(features[s:e])
        Y.append(np.bincount(labels[s:e]).argmax())
    if not X:
        return np.empty((0, window_size, features.shape[1]), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.int64)

# ── Feature builder (returns derived features + optional gps_valid) ──
def _build_session_features(
    session, builder, gnss_blackout_s=None, blackout_seed=0,
):
    """
    Returns: derived_features[N, d_derived], labels[N], gps_valid[N]|None, base_dim
    gps_valid is only returned when builder includes it in its FeatureMap specs.
    """
    gnss = session.gnss
    if gnss_blackout_s is not None and gnss_blackout_s > 0:
        gnss = apply_gnss_blackout(gnss, session.imu, gnss_blackout_s, seed=blackout_seed)

    aligner = TimeAligner(max_gap_ms=2000)
    aligned, _ = aligner.align(session.imu, gnss)

    base_features, fmap, _ = builder.build(aligned)
    clean_df = pd.DataFrame(base_features, columns=fmap.columns)

    # Only extract gps_valid if builder explicitly includes it
    has_gps_valid = "gps_valid" in fmap._index
    gpsv = base_features[:, fmap.index_of("gps_valid")].copy() if has_gps_valid else None

    if "label" in aligned.columns:
        clean_df["label"] = aligned["label"].values
    else:
        clean_df["label"] = session.label
    clean_df["label"] = clean_df["label"].str.upper().str.strip()

    extractor = FeatureExtractor(rolling_window=10)
    clean_df = extractor.transform(clean_df)

    feature_cols = [c for c in extractor.ALL_COLS if c in clean_df.columns]
    feat = clean_df[feature_cols].values.astype(np.float32)
    labels_raw = clean_df["label"].map(LABEL_MAP).values

    valid_mask = ~np.isnan(labels_raw)
    feat = feat[valid_mask]
    labels = labels_raw[valid_mask].astype(np.int64)
    if gpsv is not None:
        gpsv = gpsv[valid_mask]

    return feat, labels, gpsv, len(fmap.columns)

def build_split_dataset(
    sessions, builder, gnss_blackout_s=None, blackout_seed=0,
    window_size=150, stride=50, train_ratio=0.8,
):
    all_feat, all_labels = [], []
    for s in sessions:
        feat, lbl, gpsv, bdim = _build_session_features(
            s, builder, gnss_blackout_s, blackout_seed=blackout_seed
        )
        if gpsv is not None:
            feat = np.column_stack([feat, np.expand_dims(gpsv, 1)])
        all_feat.append(feat)
        all_labels.append(lbl)

    feat_cat = np.concatenate(all_feat, axis=0)
    lbl_cat = np.concatenate(all_labels, axis=0)
    input_dim = feat_cat.shape[1]

    n = len(feat_cat)
    n_train = int(n * train_ratio)
    idx = np.random.RandomState(42).permutation(n)
    scaler = StandardScaler()
    feat_cat = scaler.fit_transform(feat_cat).astype(np.float32)

    Xt, Yt = _sliding_windows(feat_cat[idx[:n_train]], lbl_cat[idx[:n_train]], window_size, stride)
    Xv, Yv = _sliding_windows(feat_cat[idx[n_train:]], lbl_cat[idx[n_train:]], window_size, stride)
    return Xt, Yt, Xv, Yv, input_dim

# ── Train ──
def train_model(X_train, Y_train, X_val, Y_val, input_size, epochs, batch_size, lr, device):
    model = MotionLSTM(
        input_size=input_size, hidden_size=64, num_layers=1,
        output_size=5, dropout=0.3,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(Y_train, dtype=torch.int64),
        ), batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(Y_val, dtype=torch.int64),
        ), batch_size=batch_size, shuffle=False,
    )
    history = {"train_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        history["train_loss"].append(total_loss / len(train_loader.dataset))
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        history["val_acc"].append(correct / total)
    return model, history

# ── Evaluate ──
@torch.no_grad()
def evaluate(model, X, Y, device, batch_size=64):
    model.eval()
    all_preds, all_labels = [], []
    loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.int64),
        ), batch_size=batch_size,
    )
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(yb.numpy())
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    n = len(LABEL_MAP)
    labels_arg = list(range(n))
    acc = accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average="macro", zero_division=0)
    _, _, wf1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=labels_arg)
    inv = {v: k for k, v in LABEL_MAP.items()}
    names = [inv[i] for i in range(n)]
    si, ci, cyi = LABEL_MAP["STATIONARY"], LABEL_MAP["CAR"], LABEL_MAP["CYCLING"]
    return {
        "accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1,
        "cm": cm.tolist(), "cm_labels": names,
        "sta_to_car": int(cm[si, ci]), "car_to_sta": int(cm[ci, si]),
        "sta_to_cyc": int(cm[si, cyi]), "cyc_to_sta": int(cm[cyi, si]),
        "report": classification_report(labels, preds, target_names=names, labels=labels_arg, digits=3, zero_division=0),
    }

# ── Main ──
def run_experiment(config=None):
    if config is None:
        config = ExperimentConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 70)
    print("  GNSS Blackout A/B Experiment")
    print(f"  Baseline (NaN→0, 10D→21D) vs Mask (+gps_valid, 11D→22D)")
    print("=" * 70)
    print(f"  Device: {device}  Epochs: {config.epochs}  Durations: {config.blackout_durations_s}s")

    ds = SessionDataset(config.data_dir, verbose=False)
    sessions = ds.sessions
    print(f"  Sessions: {len(sessions)}")

    builder_b = FeatureBuilder()
    builder_m = FeatureBuilder(extra_specs=[FeatureSpec(name="gps_valid", source_col="gps_valid")])

    results = []
    for dur in config.blackout_durations_s:
        print(f"\n  ── {dur:.0f}s ──")

        Xt_b, Yt_b, Xv_bc, Yv_bc, dim_b = build_split_dataset(
            sessions, builder_b, gnss_blackout_s=None,
            window_size=config.window_size, stride=config.stride, train_ratio=config.train_ratio)
        Xt_m, Yt_m, Xv_mc, Yv_mc, dim_m = build_split_dataset(
            sessions, builder_m, gnss_blackout_s=None,
            window_size=config.window_size, stride=config.stride, train_ratio=config.train_ratio)
        _, _, Xv_bd, Yv_bd, _ = build_split_dataset(
            sessions, builder_b, gnss_blackout_s=dur, blackout_seed=int(dur),
            window_size=config.window_size, stride=config.stride, train_ratio=config.train_ratio)
        _, _, Xv_md, Yv_md, _ = build_split_dataset(
            sessions, builder_m, gnss_blackout_s=dur, blackout_seed=int(dur),
            window_size=config.window_size, stride=config.stride, train_ratio=config.train_ratio)

        print(f"    Baseline: {dim_b}D, Mask: {dim_m}D")
        print(f"    Train: {len(Xt_b)} windows  Clean: {len(Xv_bc)}  Blackout: {len(Xv_bd)}")

        t0 = time.time()
        mb, hb = train_model(Xt_b, Yt_b, Xv_bd, Yv_bd, dim_b, config.epochs, config.batch_size, config.lr, device)
        tb = time.time() - t0
        t0 = time.time()
        mm, hm = train_model(Xt_m, Yt_m, Xv_md, Yv_md, dim_m, config.epochs, config.batch_size, config.lr, device)
        tm = time.time() - t0

        eb = evaluate(mb, Xv_bc, Yv_bc, device)
        em = evaluate(mm, Xv_mc, Yv_mc, device)
        eb_dur = evaluate(mb, Xv_bd, Yv_bd, device)
        em_dur = evaluate(mm, Xv_md, Yv_md, device)

        row = {
            "duration_s": dur, "input_dim_baseline": dim_b, "input_dim_mask": dim_m,
            "train_windows": len(Xt_b), "val_windows_clean": len(Xv_bc), "val_windows_blackout": len(Xv_bd),
            "epochs": config.epochs, "seed": config.seed,
            "baseline_train_s": tb, "mask_train_s": tm,
            "baseline_clean_acc": eb["accuracy"], "baseline_clean_macro_f1": eb["macro_f1"],
            "baseline_blackout_acc": eb_dur["accuracy"], "baseline_blackout_macro_f1": eb_dur["macro_f1"],
            "baseline_sta_to_car": eb_dur["sta_to_car"], "baseline_car_to_sta": eb_dur["car_to_sta"],
            "baseline_sta_to_cyc": eb_dur["sta_to_cyc"], "baseline_cyc_to_sta": eb_dur["cyc_to_sta"],
            "mask_clean_acc": em["accuracy"], "mask_clean_macro_f1": em["macro_f1"],
            "mask_blackout_acc": em_dur["accuracy"], "mask_blackout_macro_f1": em_dur["macro_f1"],
            "mask_sta_to_car": em_dur["sta_to_car"], "mask_car_to_sta": em_dur["car_to_sta"],
            "mask_sta_to_cyc": em_dur["sta_to_cyc"], "mask_cyc_to_sta": em_dur["cyc_to_sta"],
        }
        results.append(row)

        print(f"    Clean Acc:     Baseline={eb['accuracy']:.4f}  Mask={em['accuracy']:.4f}")
        print(f"    Blackout Acc:  Baseline={eb_dur['accuracy']:.4f}  Mask={em_dur['accuracy']:.4f}")

        run_file = os.path.join(config.output_dir, f"run_{int(dur)}s.json")
        with open(run_file, "w") as f:
            json.dump({"config": {k: v for k, v in vars(config).items() if k != "device"},
                       "row": row, "baseline_report": eb_dur["report"], "mask_report": em_dur["report"]},
                      f, indent=2, default=str)

    df = pd.DataFrame(results)
    csv_path = os.path.join(config.output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n{'=' * 70}")
    print(f"  Results → {csv_path}")
    cols = ["duration_s", "input_dim_baseline", "input_dim_mask",
            "baseline_blackout_acc", "mask_blackout_acc",
            "baseline_blackout_macro_f1", "mask_blackout_macro_f1",
            "baseline_car_to_sta", "mask_car_to_sta",
            "baseline_sta_to_car", "mask_sta_to_car"]
    print(df[cols].to_string(index=False))
    print("=" * 70)
    return df

if __name__ == "__main__":
    run_experiment()
