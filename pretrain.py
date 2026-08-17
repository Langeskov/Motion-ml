"""
pretrain.py V5
==============
Stage 1: 自监督预训练 — Masked Sensor Modeling。

核心思想:
  随机 mask 部分时间点的传感器数据，训练模型恢复被 mask 的值。
  无需人工标签，可使用大量无标签数据。

流程:
  1. 加载无标签 CSV 数据
  2. 随机 mask 15% 的时间步
  3. Transformer Encoder 编码
  4. Reconstruction Head 预测被 mask 的原始值
  5. 仅对 mask 位置计算 MSE Loss

输出:
  checkpoints/encoder_pretrained.pth

用法:
    python pretrain.py --data_dir data/raw
    python pretrain.py --data_dir data/raw --epochs 100 --mask_ratio 0.2
    python pretrain.py --data_dir data/raw --mask_strategy block
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from dataset.session_loader import SessionDataset
from dataset.session_dataset import MotionWindowDataset
from models.motion_encoder import build_pretrain_model
from models.mask_generator import MaskGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Motion Encoder V5")
    parser.add_argument("--data_dir", type=str, default="data/raw",
                        help="Directory containing CSV files (labels ignored)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window_size", type=int, default=150)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask_ratio", type=float, default=0.15,
                        help="Ratio of time steps to mask")
    parser.add_argument("--mask_strategy", type=str, default="random",
                        choices=["random", "block"],
                        help="Mask strategy: random or block")
    parser.add_argument("--block_size", type=int, default=10,
                        help="Block size for block mask strategy")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="runs/pretrain")
    parser.add_argument("--no_tensorboard", action="store_true")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Pretrain] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[Pretrain] Using CPU")
    return dev


def build_unlabeled_dataset(
    data_dir: str,
    window_size: int = 150,
    stride: int = 50,
):
    """
    构建无标签数据集 (V5 Session 管线)。

    流程: Session 扫描 → TimeAligner → FeatureBuilder → FeatureExtractor
          → StandardScaler → 滑动窗口
    不使用 label 列 (自监督预训练不需要标签)。

    Returns:
        X: [num_samples, window_size, input_size]  numpy array
        scaler: fitted StandardScaler
    """
    sessions = SessionDataset(data_dir)
    print(sessions.summary())

    dataset = MotionWindowDataset(
        sessions=sessions,
        window_size=window_size,
        stride=stride,
    )

    X = dataset.X
    print(f"[Pretrain] Dataset: {X.shape[0]} windows, "
          f"shape=({window_size}, {X.shape[2]})")
    print(f"[Pretrain] No labels used (self-supervised)")

    return X, dataset.scaler


def train_one_epoch(
    model: nn.Module,
    mask_gen: MaskGenerator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    训练一个 epoch。

    Returns:
        (avg_loss, mask_accuracy)
        mask_accuracy: 被 mask 位置的重建精度 (相对误差 < 10% 的比例)
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct_masked = 0
    total_masked = 0

    for (X_batch,) in loader:
        X_batch = X_batch.to(device)

        # 生成 mask
        masked_x, mask = mask_gen(X_batch)

        # 前向传播
        reconstructed, _ = model(masked_x, mask)

        # 仅对 mask 位置计算 MSE Loss
        # mask: [batch, seq_len] → [batch, seq_len, 1]
        mask_expanded = mask.unsqueeze(-1).float()

        # 计算每个位置的 MSE
        squared_error = (reconstructed - X_batch) ** 2

        # 仅对 mask 位置求平均
        masked_loss = (squared_error * mask_expanded).sum() / mask_expanded.sum()

        # 反向传播
        optimizer.zero_grad()
        masked_loss.backward()
        optimizer.step()

        total_loss += masked_loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)

        # 计算 mask 位置的重建精度 (相对误差 < 10%)
        with torch.no_grad():
            abs_error = torch.abs(reconstructed - X_batch)
            abs_target = torch.abs(X_batch) + 1e-8
            relative_error = abs_error / abs_target

            # 相对误差 < 10% 视为正确
            is_correct = (relative_error < 0.10).float() * mask_expanded
            correct_masked += is_correct.sum().item()
            total_masked += mask_expanded.sum().item()

    avg_loss = total_loss / total_samples
    mask_accuracy = correct_masked / max(1, total_masked)

    return avg_loss, mask_accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    mask_gen: MaskGenerator,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """验证，返回 (avg_loss, mask_accuracy)。"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_masked = 0
    total_masked = 0

    for (X_batch,) in loader:
        X_batch = X_batch.to(device)
        masked_x, mask = mask_gen(X_batch)
        reconstructed, _ = model(masked_x, mask)

        mask_expanded = mask.unsqueeze(-1).float()
        squared_error = (reconstructed - X_batch) ** 2
        masked_loss = (squared_error * mask_expanded).sum() / mask_expanded.sum()

        total_loss += masked_loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)

        abs_error = torch.abs(reconstructed - X_batch)
        abs_target = torch.abs(X_batch) + 1e-8
        relative_error = abs_error / abs_target
        is_correct = (relative_error < 0.10).float() * mask_expanded
        correct_masked += is_correct.sum().item()
        total_masked += mask_expanded.sum().item()

    avg_loss = total_loss / total_samples
    mask_accuracy = correct_masked / max(1, total_masked)

    return avg_loss, mask_accuracy


def main():
    args = parse_args()
    device = get_device()

    print("\n" + "=" * 60)
    print("  Stage 1: Masked Sensor Modeling Pretraining")
    print("=" * 60)

    # ── 1. 构建无标签数据集 ─────────────────────────
    X, scaler = build_unlabeled_dataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
    )
    actual_input_size = X.shape[2]

    # 划分训练/验证集
    train_size = int(len(X) * args.train_ratio)
    val_size = len(X) - train_size

    train_X = torch.tensor(X[:train_size], dtype=torch.float32)
    val_X = torch.tensor(X[train_size:], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_X), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_X), batch_size=args.batch_size, shuffle=False
    )
    print(f"[Pretrain] Split: {train_size} train / {val_size} val")

    # ── 2. 构建模型 ────────────────────────────────
    model = build_pretrain_model(
        input_size=actual_input_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── 3. Mask 生成器 ─────────────────────────────
    mask_gen = MaskGenerator(
        mask_ratio=args.mask_ratio,
        block_size=args.block_size,
        mask_strategy=args.mask_strategy,
    )
    print(f"[Pretrain] Mask strategy: {args.mask_strategy}, "
          f"ratio: {args.mask_ratio}")

    # ── 4. TensorBoard ─────────────────────────────
    writer = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=args.log_dir)
        print(f"[Pretrain] TensorBoard: {args.log_dir}")

    # ── 5. 训练循环 ────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    total_train_time = 0.0

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train MAcc':>10} | "
          f"{'Val Loss':>8} | {'Val MAcc':>8} | {'LR':>9} | {'Time':>6}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, mask_gen, train_loader, optimizer, device
        )
        val_loss, val_acc = validate(
            model, mask_gen, val_loader, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        total_train_time += elapsed

        # TensorBoard
        if writer:
            writer.add_scalars("Loss", {
                "train": train_loss, "val": val_loss
            }, epoch)
            writer.add_scalars("MaskAccuracy", {
                "train": train_acc, "val": val_acc
            }, epoch)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        print(
            f"{epoch:6d} | {train_loss:10.6f} | {train_acc:9.1%} | "
            f"{val_loss:8.6f} | {val_acc:7.1%} | "
            f"{scheduler.get_last_lr()[0]:9.2e} | {elapsed:5.1f}s"
        )

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, "encoder_pretrained.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.get_encoder_state_dict(),
                    "val_loss": val_loss,
                    "val_mask_acc": val_acc,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "input_size": actual_input_size,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"  -> Saved best encoder (val_loss={val_loss:.6f})")

    # ── 6. 保存最终模型 ────────────────────────────
    final_path = os.path.join(args.save_dir, "encoder_pretrained_final.pth")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.get_encoder_state_dict(),
            "val_loss": val_loss,
            "val_mask_acc": val_acc,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "input_size": actual_input_size,
            "args": vars(args),
        },
        final_path,
    )

    # ── 7. 保存训练历史 ────────────────────────────
    if writer:
        writer.close()

    # ── 8. 最终报告 ────────────────────────────────
    print("\n" + "=" * 60)
    print("  Pretraining Complete!")
    print("=" * 60)
    print(f"  Best Val Loss:      {best_val_loss:.6f}")
    print(f"  Total Train Time:   {total_train_time:.1f}s")
    print(f"  Avg Epoch Time:     {total_train_time / args.epochs:.1f}s")
    print(f"  Encoder saved:      {os.path.join(args.save_dir, 'encoder_pretrained.pth')}")
    print(f"  TensorBoard:        tensorboard --logdir {args.log_dir}")
    print()
    print("  Next step: Run finetune.py to fine-tune for classification")
    print("=" * 60)


if __name__ == "__main__":
    main()
