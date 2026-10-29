"""
evaluate.py
===========
评估已训练的模型在**留出会话**上的表现。

评估范围优先级:
  1. `--sessions` 显式指定的会话
  2. checkpoint 里记录的 `val_session_ids` (训练时留出的会话)

两者都没有时会报错退出 —— 不提供"在全量数据上评估"的选项,
因为全量数据包含训练样本, 得到的指标会严重虚高。
需要跨会话的整体指标请用 cv.py (会话级 K 折交叉验证)。

用法:
    python evaluate.py --checkpoint checkpoints/best_motion_encoder.pth
    python evaluate.py --checkpoint checkpoints/best_motion_encoder.pth \
        --sessions STATIONARY_20261022_100043,WALKING_20260809_180043
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from dataset.session_dataset import (
    LABEL_MAP,
    apply_scaler_to_windows,
    build_per_session_windows,
    concat_windows,
)
from models import build_model_by_type

# 反向映射: id -> name
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = [ID_TO_LABEL[i] for i in range(len(LABEL_MAP))]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Motion Model")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_motion_encoder.pth")
    parser.add_argument("--model", type=str, default=None,
                        help="强制指定模型类型 (默认从 checkpoint 读取)")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=150)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--target_hz", type=float, default=50.0)
    parser.add_argument("--min_source_hz", type=float, default=20.0)
    parser.add_argument("--drop_features", type=str, default="yaw,pitch,roll")
    parser.add_argument("--sessions", type=str, default=None,
                        help="逗号分隔的 session_id 列表 (默认读取 checkpoint 中的验证会话)")
    return parser.parse_args()


def load_checkpoint(path: str, device: torch.device, force_model_type=None) -> tuple:
    """
    加载 checkpoint，返回 (model, scaler_mean, scaler_scale, args, model_type, val_session_ids)。
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    val_session_ids = ckpt.get("val_session_ids", []) or []

    model_type = force_model_type or ckpt.get(
        "model_type", args.get("model_type", "motion_encoder"))
    input_size = ckpt.get("input_size", args.get("input_size", 24))
    hidden_size = args.get("hidden_size", 64)
    num_layers = args.get("num_layers", 2)
    output_size = args.get("output_size", len(LABEL_MAP))
    dropout = args.get("dropout", 0.3)

    model = build_model_by_type(
        model_type=model_type,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    scaler_mean = np.array(ckpt["scaler_mean"])
    scaler_scale = np.array(ckpt["scaler_scale"])

    return model, scaler_mean, scaler_scale, args, model_type, val_session_ids


def print_confusion_matrix(cm: np.ndarray, labels: list[str]):
    """格式化打印混淆矩阵。"""
    n = len(labels)
    header = "         " + "".join(f"{l:>12s}" for l in labels)
    print(header)
    print("  " + "-" * (12 * n + 8))

    for i in range(n):
        row = f"{labels[i]:>8s} |"
        for j in range(n):
            row += f"{cm[i][j]:>11d} "
        print(row)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Using {device}")

    # ── 1. 加载模型 ──────────────────────────────────
    print(f"[Eval] Loading checkpoint: {args.checkpoint}")
    model, scaler_mean, scaler_scale, train_args, model_type, ckpt_val_sessions = (
        load_checkpoint(args.checkpoint, device, args.model)
    )
    print(f"[Eval] Model type: {model_type}")
    print(f"[Eval] Parameters: {model.count_parameters():,}")

    # ── 2. 确定评估范围 ─────────────────────────────
    if args.sessions:
        eval_ids = [s.strip() for s in args.sessions.split(",") if s.strip()]
    else:
        eval_ids = ckpt_val_sessions or None

    if not eval_ids:
        raise SystemExit(
            "[Eval] 没有可评估的会话: checkpoint 里没有 val_session_ids,"
            " 也没传 --sessions。\n"
            "       本工具只评估留出会话, 不提供全量评估 —— 全量数据包含训练样本,\n"
            "       得到的指标会严重虚高。需要整体指标请用 cv.py。"
        )

    drop_features = [s.strip() for s in args.drop_features.split(",") if s.strip()]
    print(f"\n[Eval] 评估留出会话 ({len(eval_ids)} 个, 模型训练时未见过):")
    for s in eval_ids:
        print(f"         - {s}")

    per_session = build_per_session_windows(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
        drop_features=drop_features,
        target_hz=args.target_hz,
        min_source_hz=args.min_source_hz,
    )
    known = {p["session_id"] for p in per_session}
    missing = [s for s in eval_ids if s not in known]
    if missing:
        raise SystemExit(f"[Eval] 会话在数据目录中不存在: {missing}")

    want = set(eval_ids)
    picked = [p for p in per_session if p["session_id"] in want]
    X_raw, Y = concat_windows(picked)

    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    scaler.var_ = scaler_scale ** 2
    scaler.n_features_in_ = len(scaler_mean)
    X = apply_scaler_to_windows(scaler, X_raw)

    eval_set = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    print(f"[Eval] 评估样本数: {len(X):,}")
    loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)

    # ── 3. 推理 ──────────────────────────────────────
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, Y_batch in loader:
            logits = model(X_batch.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(Y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── 4. 输出评估结果 ──────────────────────────────
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    print("\n" + "=" * 60)
    print(f"  Evaluation Results: {model_type.upper()}")
    print("=" * 60)

    print(f"\n  Overall Accuracy:  {accuracy:.1%}")
    print(f"  Weighted Precision: {precision:.3f}")
    print(f"  Weighted Recall:    {recall:.3f}")
    print(f"  Weighted F1-score:  {f1:.3f}")
    print(f"  Total Samples:     {len(all_labels)}")

    cm = confusion_matrix(all_labels, all_preds,
                          labels=list(range(len(LABEL_MAP))))
    print("\n  Confusion Matrix:")
    print_confusion_matrix(cm, LABEL_NAMES)

    print("\n  Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=LABEL_NAMES, digits=3, zero_division=0,
    ))

    print("=" * 60)


if __name__ == "__main__":
    main()
