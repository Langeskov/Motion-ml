"""
evaluate.py V5
==============
评估已训练的模型 (支持 LSTM、CNN-LSTM、CNN-Transformer、MotionEncoder)。

V5 变更:
  - 支持 motion_encoder / motion_classifier 模型类型
  - 自动从 checkpoint 读取 model_type

输出:
    - Accuracy
    - Precision / Recall / F1-score
    - Confusion Matrix
    - Classification Report

用法:
    python evaluate.py --checkpoint checkpoints/best_lstm.pth
    python evaluate.py --checkpoint checkpoints/motion_classifier.pth
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from dataset.loader import build_dataset, LABEL_MAP
from models import load_model_from_checkpoint, build_model_by_type
from models.motion_encoder import build_classifier

# 反向映射: id -> name
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
LABEL_NAMES = [ID_TO_LABEL[i] for i in range(len(LABEL_MAP))]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Motion Model V5")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_lstm.pth")
    parser.add_argument("--model", type=str, default=None,
                        help="Force model type (auto-detected from checkpoint if omitted)")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=150)
    parser.add_argument("--stride", type=int, default=50)
    return parser.parse_args()


def load_checkpoint(path: str, device: torch.device, force_model_type=None) -> tuple:
    """
    加载 checkpoint，返回 (model, scaler_mean, scaler_scale, args, model_type)。
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})

    model_type = force_model_type or ckpt.get("model_type", args.get("model_type", "lstm"))
    input_size = ckpt.get("input_size", args.get("input_size", 21))
    hidden_size = args.get("hidden_size", 128)
    num_layers = args.get("num_layers", 2)
    output_size = args.get("output_size", len(LABEL_MAP))
    dropout = args.get("dropout", 0.3)

    # 特殊处理 motion_classifier checkpoint
    if model_type == "motion_classifier":
        model = build_classifier(
            input_size=input_size,
            d_model=hidden_size,
            nhead=4,
            num_layers=num_layers,
            num_classes=output_size,
            dropout=dropout,
        )
    else:
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

    return model, scaler_mean, scaler_scale, args, model_type


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
    model, scaler_mean, scaler_scale, train_args, model_type = load_checkpoint(
        args.checkpoint, device, args.model
    )
    print(f"[Eval] Model type: {model_type}")
    print(f"[Eval] Parameters: {model.count_parameters():,}")

    # ── 2. 构建数据集 (使用训练时的 scaler) ─────────
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale

    dataset, _ = build_dataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
        scaler=scaler,
        fit_scaler=False,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # ── 3. 推理 ──────────────────────────────────────
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
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

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(LABEL_MAP))))
    print("\n  Confusion Matrix:")
    print_confusion_matrix(cm, LABEL_NAMES)

    # Classification Report
    print("\n  Classification Report:")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=LABEL_NAMES,
        digits=3,
        zero_division=0,
    )
    print(report)

    print("=" * 60)


if __name__ == "__main__":
    main()
