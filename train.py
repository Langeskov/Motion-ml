"""
train.py V4
============
Motion 模型训练脚本，支持 LSTM、CNN-LSTM、CNN-Transformer 三种模型。

V4 变更:
  - --model 参数新增: cnn_transformer
  - 支持保存 attention weights (CNN-Transformer)
  - 训练结束后输出完整评估指标 (accuracy, precision, recall, F1, confusion matrix)

用法:
    python train.py --model lstm
    python train.py --model cnn_lstm
    python train.py --model cnn_transformer
    python train.py --model cnn_transformer --epochs 100 --lr 0.0005
    python train.py --model cnn_transformer --save_attn
    tensorboard --logdir runs/
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
from models import build_model_by_type, MODEL_TYPES


def parse_args():
    parser = argparse.ArgumentParser(description="Train Motion Model V4")
    parser.add_argument("--model", type=str, default="lstm",
                        choices=sorted(MODEL_TYPES),
                        help="Model type: lstm, cnn_lstm, or cnn_transformer")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window_size", type=int, default=150)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--split_mode", type=str, default="session",
                        choices=["session", "window"],
                        help="session=留出整段会话(可信评测, 默认); "
                             "window=窗口级随机划分(存在时间泄漏, 仅用于冒烟测试)")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--select_metric", type=str, default="acc",
                        choices=["acc", "macro_f1"],
                        help="选择最优模型的指标。数据不均衡时建议 macro_f1"
                             " (accuracy 会被占比最大的类主导)")
    parser.add_argument("--class_weight", action="store_true",
                        help="按类别频率反比加权 CrossEntropyLoss (缓解类别不均衡)")
    parser.add_argument("--drop_features", type=str, default="",
                        help="逗号分隔的待剔除特征名, 如 'yaw,pitch,roll'。"
                             "绝对姿态角依赖手机朝向, 跨会话泛化差")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="TensorBoard log dir (default: runs/<model_type>)")
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--save_attn", action="store_true",
                        help="Save attention weights (cnn_transformer only)")
    return parser.parse_args()


def get_device() -> torch.device:
    """自动检测 CUDA，回退到 CPU。"""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Train] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[Train] Using CPU")
    return dev


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """训练一个 epoch，返回 (平均loss, accuracy)。"""
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

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    验证，返回 (平均loss, accuracy, all_preds, all_labels)。
    """
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

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def save_attention_weights(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_dir: str,
    model_type: str,
    max_batches: int = 5,
):
    """
    保存 CNN-Transformer 的 attention weights。

    Args:
        model: CNNTransformer 模型
        loader: 数据加载器
        device: 设备
        save_dir: 保存目录
        model_type: 模型类型
        max_batches: 最多保存的 batch 数
    """
    if not hasattr(model, "get_attention_weights"):
        print("[Train] Model does not support attention weights")
        return

    model.eval()
    all_weights = []
    all_labels = []

    with torch.no_grad():
        for i, (X_batch, Y_batch) in enumerate(loader):
            if i >= max_batches:
                break
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            weights = model.get_attention_weights()
            if weights is not None:
                all_weights.append(weights.cpu().numpy())
                all_labels.append(Y_batch.numpy())

    if all_weights:
        weights_array = np.concatenate(all_weights, axis=0)
        labels_array = np.concatenate(all_labels, axis=0)

        attn_path = os.path.join(save_dir, f"attention_weights_{model_type}.npz")
        np.savez(
            attn_path,
            weights=weights_array,
            labels=labels_array,
        )
        print(f"[Train] Saved attention weights: {attn_path}")
        print(f"  Shape: {weights_array.shape}")


def log_confusion_matrix(
    writer: SummaryWriter,
    preds: np.ndarray,
    labels: np.ndarray,
    epoch: int,
    num_classes: int = 5,
):
    """将混淆矩阵记录到 TensorBoard (作为图像)。"""
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


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    计算完整评估指标。

    Returns:
        dict with accuracy, precision, recall, f1, confusion_matrix, classification_report
    """
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


def print_final_report(
    model_type: str,
    model: nn.Module,
    best_val_acc: float,
    total_train_time: float,
    epochs: int,
    val_preds: np.ndarray,
    val_labels: np.ndarray,
):
    """训练结束后打印完整模型报告。"""
    param_count = model.count_parameters()
    metrics = compute_metrics(val_preds, val_labels)

    print("\n" + "=" * 60)
    print(f"  Training Report: {model_type.upper()}")
    print("=" * 60)
    print(f"  Model Type:       {model_type}")
    print(f"  Parameters:       {param_count:,}")
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

    # ── 1. 构建数据集 & 2. 划分训练/验证集 ─────────────
    print("\n" + "=" * 60)
    print(f"  Motion Model Training V5 ({args.model.upper()})")
    print("=" * 60)

    drop_features = [s.strip() for s in args.drop_features.split(",") if s.strip()]

    if args.split_mode == "session":
        # 可信协议: 留出整段会话, 验证集与训练集无时间重叠
        per_session = build_per_session_windows(
            data_dir=args.data_dir,
            window_size=args.window_size,
            stride=args.stride,
            drop_features=drop_features,
        )
        train_parts, val_parts, val_session_ids = session_level_split(
            per_session,
            val_ratio=1.0 - args.train_ratio,
            seed=42,
        )
        trX_raw, trY = concat_windows(train_parts)
        vaX_raw, vaY = concat_windows(val_parts)

        # 标准化参数只在训练会话上拟合
        scaler = fit_scaler_on_windows(trX_raw)
        trX = apply_scaler_to_windows(scaler, trX_raw)
        vaX = apply_scaler_to_windows(scaler, vaX_raw)
        del trX_raw, vaX_raw

        actual_input_size = trX.shape[2]
        train_set = TensorDataset(torch.from_numpy(trX), torch.from_numpy(trY))
        val_set = TensorDataset(torch.from_numpy(vaX), torch.from_numpy(vaY))

        print(f"[Train] Split mode: session (无时间泄漏)")
        print(f"[Train] 训练: {len(train_parts)} sessions / {len(trX):,} windows")
        print(f"[Train] 验证: {len(val_parts)} sessions / {len(vaX):,} windows")
        print(f"[Train] 验证会话: {', '.join(val_session_ids)}")
    else:
        # 兼容旧行为 (窗口级 random_split, 存在 ~99.8% 泄漏, 仅用于冒烟测试)
        dataset, scaler = build_session_dataset(
            data_dir=args.data_dir,
            window_size=args.window_size,
            stride=args.stride,
            drop_features=drop_features,
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
        print(f"[Train] Split mode: window (警告: 验证集与训练集时间重叠)")
        print(f"[Train] Split: {train_size} train / {val_size} val")

    print(f"[Train] Detected input_size: {actual_input_size}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # ── 3. 构建模型 ──────────────────────────────────
    model = build_model_by_type(
        model_type=args.model,
        input_size=actual_input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=len(LABEL_MAP),
        dropout=args.dropout,
    ).to(device)

    # 损失函数: 可选类别加权 (缓解 CAR 占比 55% 的失衡)
    if args.class_weight:
        if args.split_mode == "session":
            counts = np.bincount(trY, minlength=len(LABEL_MAP)).astype(np.float64)
        else:
            counts = np.bincount(dataset.Y, minlength=len(LABEL_MAP)).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (len(LABEL_MAP) * counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device)
        )
        inv_map = {v: k for k, v in LABEL_MAP.items()}
        print("[Train] Class weights: " + ", ".join(
            f"{inv_map[i]}={weights[i]:.2f}" for i in range(len(LABEL_MAP))))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── 4. TensorBoard ───────────────────────────────
    log_dir = args.log_dir or f"runs/{args.model}"
    writer = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=log_dir)
        dummy_input = torch.randn(1, args.window_size, actual_input_size).to(device)
        # torch.jit.trace 的 check_trace 在 Transformer 模型上因图结构差异误报，
        # 临时 patch 强制 check_trace=False 绕过
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
        print(f"[Train] TensorBoard logging to: {log_dir}")

    # ── 5. 训练循环 ──────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_acc = 0.0
    best_score = -1.0
    best_preds, best_labels = None, None
    total_train_time = 0.0

    def _score(preds, labels):
        if args.select_metric == "macro_f1":
            from sklearn.metrics import f1_score
            return float(f1_score(labels, preds, average="macro", zero_division=0))
        return float((np.asarray(preds) == np.asarray(labels)).mean())

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'Time':>6}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - t0
        total_train_time += elapsed

        score = _score(val_preds, val_labels)

        # ── TensorBoard 日志 ─────────────────────────
        if writer:
            writer.add_scalars("Loss", {
                "train": train_loss,
                "val": val_loss,
            }, epoch)
            writer.add_scalars("Accuracy", {
                "train": train_acc,
                "val": val_acc,
            }, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            if epoch % 5 == 0 or epoch == args.epochs:
                log_confusion_matrix(
                    writer, val_preds, val_labels, epoch, len(LABEL_MAP)
                )

        print(
            f"{epoch:6d} | {train_loss:10.4f} | {train_acc:8.1%} | "
            f"{val_loss:8.4f} | {val_acc:6.1%} | {elapsed:5.1f}s"
        )

        # 保存最优模型 (按 select_metric 选择)
        if score > best_score:
            best_score = score
            best_val_acc = val_acc
            best_preds, best_labels = val_preds, val_labels
            save_path = os.path.join(args.save_dir, f"best_{args.model}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_type": args.model,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
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
            print(f"  -> Saved best model (val_acc={val_acc:.1%})")

    # ── 6. 保存最终模型 ──────────────────────────────
    final_path = os.path.join(args.save_dir, f"final_{args.model}.pth")
    torch.save(
        {
            "epoch": args.epochs,
            "model_type": args.model,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "input_size": actual_input_size,
            "args": vars(args),
        },
        final_path,
    )

    # ── 7. 保存 attention weights (CNN-Transformer) ──
    if args.save_attn and args.model == "cnn_transformer":
        save_attention_weights(
            model, val_loader, device, args.save_dir, args.model
        )

    # ── 8. 关闭 TensorBoard ─────────────────────────
    if writer:
        writer.close()

    # ── 9. 打印最终报告 ──────────────────────────────
    metrics = print_final_report(
        model_type=args.model,
        model=model,
        best_val_acc=best_val_acc,
        total_train_time=total_train_time,
        epochs=args.epochs,
        val_preds=best_preds if best_preds is not None else val_preds,
        val_labels=best_labels if best_labels is not None else val_labels,
    )

    # 保存 metrics 到 JSON
    metrics_path = os.path.join(args.save_dir, f"metrics_{args.model}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    print(f"  Best model: {os.path.join(args.save_dir, f'best_{args.model}.pth')}")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
