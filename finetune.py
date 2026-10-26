"""
finetune.py V5
==============
Stage 2: 有监督微调 — 加载预训练 Encoder + 分类头。

核心思想:
  加载 Stage 1 预训练的 Encoder 权重，添加分类头，
  使用有标签数据进行有监督微调。

流程:
  1. 加载有标签 CSV 数据
  2. 加载预训练 Encoder 权重
  3. 添加 AttentionPooling + ClassificationHead
  4. 先冻结 Encoder 训练分类头 (warm-up)
  5. 解冻 Encoder 端到端微调
  6. 保存最终分类模型

输出:
  checkpoints/motion_classifier.pth

用法:
    python finetune.py --data_dir data/raw --pretrained checkpoints/encoder_pretrained.pth
    python finetune.py --data_dir data/raw --pretrained checkpoints/encoder_pretrained.pth --epochs 100
    python finetune.py --data_dir data/raw  # 不使用预训练 (从零训练)
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset.session_dataset import (
    build_session_dataset,
    build_per_session_windows,
    session_level_split,
    concat_windows,
    fit_scaler_on_windows,
    apply_scaler_to_windows,
    LABEL_MAP,
)
from models.motion_encoder import build_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Motion Classifier V5")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained encoder checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Epochs to train with encoder frozen")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=1e-4,
                        help="Learning rate after unfreezing encoder")
    parser.add_argument("--window_size", type=int, default=150)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--split_mode", type=str, default="session",
                        choices=["session", "window"],
                        help="session=留出整段会话(可信评测, 默认); "
                             "window=窗口级随机划分(存在时间泄漏)")
    parser.add_argument("--target_hz", type=float, default=50.0,
                        help="重采样目标频率 Hz (默认 50); 填 0 关闭")
    parser.add_argument("--min_source_hz", type=float, default=20.0,
                        help="原生采样率低于此值的会话直接跳过")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--select_metric", type=str, default="acc",
                        choices=["acc", "macro_f1"],
                        help="选择最优模型的指标 (数据不均衡时建议 macro_f1)")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--no_tensorboard", action="store_true")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Finetune] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[Finetune] Using CPU")
    return dev


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, Y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == Y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, Y_batch)

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == Y_batch).sum().item()
        total += X_batch.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(Y_batch.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(LABEL_MAP))))
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    label_names = [inv_map[i] for i in range(len(LABEL_MAP))]
    report = classification_report(
        labels, preds, target_names=label_names, digits=3, zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def log_confusion_matrix(
    writer: SummaryWriter,
    preds: np.ndarray,
    labels: np.ndarray,
    epoch: int,
    num_classes: int = 5,
):
    from sklearn.metrics import confusion_matrix as sk_cm
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = sk_cm(labels, preds, labels=list(range(num_classes)))
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    names = [inv_map[i] for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(names)

    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    writer.add_figure("Confusion Matrix", fig, global_step=epoch)
    plt.close(fig)


def print_final_report(
    model: nn.Module,
    best_val_acc: float,
    total_train_time: float,
    epochs: int,
    val_preds: np.ndarray,
    val_labels: np.ndarray,
):
    metrics = compute_metrics(val_preds, val_labels)

    print("\n" + "=" * 60)
    print("  Fine-tuning Report")
    print("=" * 60)
    print(f"  Parameters:       {model.count_parameters():,}")
    print(f"  Best Val Acc:     {best_val_acc:.1%}")
    print(f"  Total Train Time: {total_train_time:.1f}s")
    print(f"  Avg Epoch Time:   {total_train_time / epochs:.1f}s")
    print()
    print(f"  Accuracy:         {metrics['accuracy']:.3f}")
    print(f"  Precision:        {metrics['precision']:.3f}")
    print(f"  Recall:           {metrics['recall']:.3f}")
    print(f"  F1-score:         {metrics['f1']:.3f}")
    print()
    print("  Confusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    label_names = [inv_map[i] for i in range(len(LABEL_MAP))]
    header = "         " + "".join(f"{l:>12s}" for l in label_names)
    print(header)
    print("  " + "-" * (12 * len(label_names) + 8))
    for i in range(len(label_names)):
        row = f"{label_names[i]:>8s} |"
        for j in range(len(label_names)):
            row += f"{cm[i][j]:>11d} "
        print(row)
    print()
    print("  Classification Report:")
    print(metrics["classification_report"])
    print("=" * 60)

    return metrics


def main():
    args = parse_args()
    device = get_device()

    print("\n" + "=" * 60)
    print("  Stage 2: Fine-tuning Motion Classifier")
    print("=" * 60)

    # ── 1. 构建有标签数据集 ─────────────────────────
    if args.split_mode == "session":
        per_session = build_per_session_windows(
            data_dir=args.data_dir,
            window_size=args.window_size,
            stride=args.stride,
            target_hz=args.target_hz,
            min_source_hz=args.min_source_hz,
        )
        train_parts, val_parts, val_session_ids = session_level_split(
            per_session, val_ratio=1.0 - args.train_ratio, seed=42,
        )
        trX_raw, trY = concat_windows(train_parts)
        vaX_raw, vaY = concat_windows(val_parts)

        scaler = fit_scaler_on_windows(trX_raw)
        trX = apply_scaler_to_windows(scaler, trX_raw)
        vaX = apply_scaler_to_windows(scaler, vaX_raw)
        del trX_raw, vaX_raw

        actual_input_size = trX.shape[2]
        train_set = TensorDataset(torch.from_numpy(trX), torch.from_numpy(trY))
        val_set = TensorDataset(torch.from_numpy(vaX), torch.from_numpy(vaY))

        print(f"[Finetune] Split mode: session (无时间泄漏)")
        print(f"[Finetune] 训练: {len(train_parts)} sessions / {len(trX):,} windows")
        print(f"[Finetune] 验证: {len(val_parts)} sessions / {len(vaX):,} windows")
        print(f"[Finetune] 验证会话: {', '.join(val_session_ids)}")
    else:
        dataset, scaler = build_session_dataset(
            data_dir=args.data_dir,
            window_size=args.window_size,
            stride=args.stride,
        )
        actual_input_size = dataset.X.shape[2]

        train_size = int(len(dataset) * args.train_ratio)
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        val_session_ids = []
        print(f"[Finetune] Split mode: window (警告: 验证集与训练集时间重叠)")
        print(f"[Finetune] Split: {train_size} train / {val_size} val")

    print(f"[Finetune] Detected input_size: {actual_input_size}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False
    )

    # ── 2. 构建分类模型 ────────────────────────────
    model = build_classifier(
        input_size=actual_input_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_classes=len(LABEL_MAP),
        dropout=args.dropout,
        pretrained_encoder_path=args.pretrained,
    ).to(device)

    if args.pretrained:
        print(f"[Finetune] Using pretrained encoder: {args.pretrained}")
    else:
        print("[Finetune] No pretrained encoder (training from scratch)")

    # ── 3. TensorBoard ─────────────────────────────
    log_dir = args.log_dir or f"runs/finetune"
    writer = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=log_dir)
        dummy_input = torch.randn(1, args.window_size, actual_input_size).to(device)
        import functools
        from unittest.mock import patch as _patch
        _orig_trace = torch.jit.trace
        @functools.wraps(_orig_trace)
        def _nocheck_trace(*a, **kw):
            kw["check_trace"] = False
            return _orig_trace(*a, **kw)
        model.eval()
        with _patch.object(torch.jit, "trace", _nocheck_trace):
            writer.add_graph(model, dummy_input)
        model.train()
        print(f"[Finetune] TensorBoard: {log_dir}")

    # ── 4. 训练循环 ────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    best_score = -1.0
    best_preds, best_labels = None, None
    total_train_time = 0.0
    criterion = nn.CrossEntropyLoss()

    def _score(preds, labels):
        if preds is None:
            return None
        if args.select_metric == "macro_f1":
            from sklearn.metrics import f1_score
            return float(f1_score(labels, preds, average="macro", zero_division=0))
        return float((np.asarray(preds) == np.asarray(labels)).mean())

    # Phase 1: 冻结 Encoder，仅训练分类头
    if args.pretrained and args.warmup_epochs > 0:
        print(f"\n[Finetune] Phase 1: Warm-up ({args.warmup_epochs} epochs, encoder frozen)")
        model.freeze_encoder()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
        )

        print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
              f"{'Val Loss':>8} | {'Val Acc':>7} | {'Time':>6}")
        print("-" * 65)

        for epoch in range(1, args.warmup_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, _, _ = validate(
                model, val_loader, criterion, device
            )
            elapsed = time.time() - t0
            total_train_time += elapsed

            if writer:
                writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
                writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)

            print(
                f"{epoch:6d} | {train_loss:10.4f} | {train_acc:8.1%} | "
                f"{val_loss:8.4f} | {val_acc:6.1%} | {elapsed:5.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc

    # Phase 2: 解冻 Encoder，端到端微调
    print(f"\n[Finetune] Phase 2: Fine-tuning ({args.epochs} epochs, encoder unfrozen)")
    model.unfreeze_encoder()
    # Phase 2 重新开始记录最优 (warmup 的指标不作为选拔基准)
    best_val_acc = 0.0
    best_score = -1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>9} | {'Time':>6}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        total_train_time += elapsed

        if writer:
            writer.add_scalars("Loss_FT", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("Accuracy_FT", {"train": train_acc, "val": val_acc}, epoch)
            writer.add_scalar("LR_FT", scheduler.get_last_lr()[0], epoch)

            if epoch % 5 == 0 or epoch == args.epochs:
                log_confusion_matrix(
                    writer, val_preds, val_labels, epoch, len(LABEL_MAP)
                )

        print(
            f"{epoch:6d} | {train_loss:10.4f} | {train_acc:8.1%} | "
            f"{val_loss:8.4f} | {val_acc:6.1%} | "
            f"{scheduler.get_last_lr()[0]:9.2e} | {elapsed:5.1f}s"
        )

        # 保存最优模型 (按 select_metric 选择)
        score = _score(val_preds, val_labels)
        if score > best_score:
            best_score = score
            best_val_acc = val_acc
            best_preds, best_labels = val_preds, val_labels
            save_path = os.path.join(args.save_dir, "motion_classifier.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_type": "motion_classifier",
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "input_size": actual_input_size,
                    "split_mode": args.split_mode,
                    "val_session_ids": val_session_ids,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"  -> Saved best classifier (val_acc={val_acc:.1%})")

    # ── 5. 保存最终模型 ────────────────────────────
    final_path = os.path.join(args.save_dir, "motion_classifier_final.pth")
    torch.save(
        {
            "epoch": args.epochs,
            "model_type": "motion_classifier",
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "val_acc": val_acc,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "input_size": actual_input_size,
            "split_mode": args.split_mode,
            "val_session_ids": val_session_ids,
            "args": vars(args),
        },
        final_path,
    )

    # ── 6. 关闭 TensorBoard ───────────────────────
    if writer:
        writer.close()

    # ── 7. 打印最终报告 ────────────────────────────
    metrics = print_final_report(
        model=model,
        best_val_acc=best_val_acc,
        total_train_time=total_train_time,
        epochs=args.epochs,
        val_preds=best_preds if best_preds is not None else val_preds,
        val_labels=best_labels if best_labels is not None else val_labels,
    )

    # 保存 metrics
    metrics_path = os.path.join(args.save_dir, "metrics_finetune.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    print(f"  Best model: {os.path.join(args.save_dir, 'motion_classifier.pth')}")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
