"""
cv.py — Session 级 K 折交叉验证
================================

为什么需要这个:
  数据采集成本高, 每类只有 2~9 个会话。用单次"留出几个会话"评估时,
  macro F1 里每一类其实只由 1 个会话决定, 换个划分就从 0.44 跳到 0.77,
  结论完全不可靠 (实测同配置换种子, acc 波动 ±3.5pt)。

  改成按会话分 K 折: 每类会话轮流做验证集, 所有会话都被评到恰好一次,
  汇总袋外 (out-of-fold) 预测后, 每一类的指标都建立在**该类的全部会话**上,
  方差大幅下降。这是在"不新增采集"前提下能得到的最稳评估。

用法:
    python cv.py --folds 3
    python cv.py --folds 3 --model cnn_transformer --epochs 60 --target_hz 50

输出:
    - 每折的验证会话与指标
    - 袋外汇总: accuracy / macro F1 / 每类 F1 / 混淆矩阵
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, TensorDataset

from dataset.session_dataset import (
    LABEL_MAP,
    apply_scaler_to_windows,
    build_per_session_windows,
    concat_windows,
    fit_scaler_on_windows,
)
from models import MODEL_TYPES, build_model_by_type
from train import ModelEMA, build_scheduler, train_one_epoch, validate

INV = {v: k for k, v in LABEL_MAP.items()}


def parse_args():
    p = argparse.ArgumentParser(description="Session-level K-fold CV")
    p.add_argument("--model", type=str, default="cnn_transformer",
                   choices=sorted(MODEL_TYPES))
    p.add_argument("--data_dir", type=str, default="data/raw")
    p.add_argument("--folds", type=int, default=3,
                   help="折数 (会被每类会话数限制, 保证每折都能分到该类会话)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int, default=64,
                   help="隐藏层维度。默认 64: 128 在本数据规模下会过拟合")
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--window_size", type=int, default=150)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument("--target_hz", type=float, default=50.0)
    p.add_argument("--min_source_hz", type=float, default=20.0)
    p.add_argument("--drop_features", type=str, default="yaw,pitch,roll")
    p.add_argument("--select_metric", type=str, default="macro_f1",
                   choices=["acc", "macro_f1"])
    p.add_argument("--min_epochs", type=int, default=15,
                   help="至少训练这么多轮才允许早停 (0=不限)")
    p.add_argument("--lr_schedule", type=str, default="cosine",
                   choices=["none", "cosine"],
                   help="学习率调度: cosine=warmup+余弦衰减(默认); none=恒定 lr")
    p.add_argument("--ema", action="store_true",
                   help="权重指数滑动平均, 用 EMA 权重做验证/保存")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--augment", action="store_true",
                   help="训练数据增强 (幅度缩放+高斯抖动+时间平移)")
    p.add_argument("--aug_jitter", type=float, default=0.01)
    p.add_argument("--aug_shift", type=int, default=5)
    p.add_argument("--exclude_sessions", type=str, default="",
                   help="逗号分隔的会话 id, 直接排除 (用于剔除标签可疑/数据异常的会话)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="cv_result.json")
    return p.parse_args()


def assign_folds(per_session, k, seed):
    """按类别分层、类内确定性轮转分配折号, 保证每折都覆盖每个类别。"""
    by_label = {}
    for p in per_session:
        by_label.setdefault(p["label_id"], []).append(p)

    fold_of = {}
    for label_id, group in by_label.items():
        ordered = sorted(group, key=lambda x: x["session_id"])
        rng = np.random.RandomState(seed + label_id)
        perm = rng.permutation(len(ordered))
        for pos, idx in enumerate(perm):
            fold_of[ordered[idx]["session_id"]] = pos % k
    return fold_of


def train_fold(args, tr_parts, va_parts, device, fold_no, k):
    trX_raw, trY = concat_windows(tr_parts)
    vaX_raw, vaY = concat_windows(va_parts)

    scaler = fit_scaler_on_windows(trX_raw)     # 只在训练折上拟合
    trX = apply_scaler_to_windows(scaler, trX_raw)
    vaX = apply_scaler_to_windows(scaler, vaX_raw)
    del trX_raw, vaX_raw

    train_set = TensorDataset(torch.from_numpy(trX), torch.from_numpy(trY))
    val_set = TensorDataset(torch.from_numpy(vaX), torch.from_numpy(vaY))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    torch.manual_seed(args.seed + fold_no)
    np.random.seed(args.seed + fold_no)

    model = build_model_by_type(
        model_type=args.model,
        input_size=trX.shape[2],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=len(LABEL_MAP),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer, max(1, len(train_loader)), args.epochs, args.lr_schedule
    )
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    best_score, best_state, best_pred, best_true = -1.0, None, None, None
    best_epoch, no_improve = 0, 0

    def score_of(pred, true):
        if args.select_metric == "macro_f1":
            return float(f1_score(true, pred, average="macro", zero_division=0))
        return float((np.asarray(pred) == np.asarray(true)).mean())

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device,
                        scheduler=scheduler, ema=ema,
                        augment=args.augment,
                        aug_jitter=args.aug_jitter, aug_shift=args.aug_shift)

        # 用 EMA 权重评估 (EMA 已在每个 step 更新过), 验证后换回原始权重继续训练
        if ema is not None:
            raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state_dict())

        _, _, pred, true = validate(model, val_loader, criterion, device)
        eval_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if ema is not None:
            model.load_state_dict(raw_state)

        s = score_of(pred, true)
        if s > best_score:
            best_score, best_epoch, no_improve = s, epoch, 0
            best_pred, best_true = pred, true
            best_state = eval_state
        else:
            no_improve += 1
            if (args.early_stop_patience > 0
                    and no_improve >= args.early_stop_patience
                    and epoch >= args.min_epochs):
                break
    dt = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "fold": fold_no,
        "val_sessions": [p["session_id"] for p in va_parts],
        "pred": best_pred,
        "true": best_true,
        "score": best_score,
        "best_epoch": best_epoch,
        "ran_epochs": epoch,
        "secs": dt,
        "n_train": len(trX),
        "n_val": len(vaX),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CV] device={device}")

    drop_features = [s.strip() for s in args.drop_features.split(",") if s.strip()]
    exclude = [s.strip() for s in args.exclude_sessions.split(",") if s.strip()]
    print("[CV] 构建逐会话窗口 ...")
    per = build_per_session_windows(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
        drop_features=drop_features,
        target_hz=args.target_hz,
        min_source_hz=args.min_source_hz,
        exclude_sessions=exclude or None,
    )

    counts = {}
    for p in per:
        counts.setdefault(p["label"], 0)
        counts[p["label"]] += 1
    print(f"[CV] {len(per)} 个会话, 每类会话数: {dict(sorted(counts.items()))}")

    k = args.folds
    min_cls = min(counts.values())
    if k > min_cls:
        print(f"[CV] folds 从 {k} 收敛到 {min_cls} (受最小类 {min_cls} 个会话限制)")
        k = min_cls
    if k < 2:
        raise SystemExit("[CV] 至少需要 2 折")

    fold_of = assign_folds(per, k, args.seed)
    print(f"[CV] {k} 折分配:")
    for f in range(k):
        ids = [p["session_id"] for p in per if fold_of[p["session_id"]] == f]
        print(f"   fold {f}: {len(ids)} 个会话")
        for s in ids:
            print(f"      {s}")

    results = []
    for f in range(k):
        tr = [p for p in per if fold_of[p["session_id"]] != f]
        va = [p for p in per if fold_of[p["session_id"]] == f]
        r = train_fold(args, tr, va, device, f, k)
        results.append(r)
        acc = accuracy_score(r["true"], r["pred"])
        mac = f1_score(r["true"], r["pred"], average="macro", zero_division=0)
        print(f"\n[CV] fold {f}: acc={acc:.3f} macroF1={mac:.3f} "
              f"(best epoch {r['best_epoch']}, 跑了 {r['ran_epochs']} 轮, "
              f"{r['secs']:.0f}s, {r['n_train']}/{r['n_val']} 窗口)")
        for s in r["val_sessions"]:
            print(f"      val: {s}")

    # ── 袋外汇总 ──
    oof_pred = np.concatenate([r["pred"] for r in results])
    oof_true = np.concatenate([r["true"] for r in results])

    acc = accuracy_score(oof_true, oof_pred)
    macro = f1_score(oof_true, oof_pred, average="macro", zero_division=0)
    per_cls = precision_recall_fscore_support(
        oof_true, oof_pred, labels=list(range(len(LABEL_MAP))),
        average=None, zero_division=0)[2]
    cm = confusion_matrix(oof_true, oof_pred, labels=list(range(len(LABEL_MAP))))

    print("\n" + "=" * 74)
    print(f"  袋外 (out-of-fold) 汇总 — {k} 折, 每类会话全部参与评估")
    print("=" * 74)
    print(f"  样本数: {len(oof_true):,}")
    print(f"  Accuracy: {acc:.4f}    Macro F1: {macro:.4f}")
    print(f"\n  {'类别':<12s}{'F1':>8s}{'support':>10s}")
    print("  " + "-" * 30)
    for i in range(len(LABEL_MAP)):
        sup = int((oof_true == i).sum())
        print(f"  {INV[i]:<12s}{per_cls[i]:>8.3f}{sup:>10d}")

    print(f"\n  混淆矩阵 (行=真实, 列=预测):")
    print("        " + "".join(f"{INV[j]:>12s}" for j in range(len(LABEL_MAP))))
    for i in range(len(LABEL_MAP)):
        print(f"  {INV[i]:>6s}" + "".join(f"{cm[i][j]:>12d}" for j in range(len(LABEL_MAP))))

    print(f"\n  每折最优 epoch: {[r['best_epoch'] for r in results]}")
    print(f"  每折实际轮数  : {[r['ran_epochs'] for r in results]}")
    print(f"  总耗时        : {sum(r['secs'] for r in results):.0f}s")

    out = {
        "folds": k,
        "args": vars(args),
        "oof_accuracy": float(acc),
        "oof_macro_f1": float(macro),
        "per_class_f1": {INV[i]: float(per_cls[i]) for i in range(len(LABEL_MAP))},
        "confusion_matrix": cm.tolist(),
        "per_fold": [{
            "fold": r["fold"], "val_sessions": r["val_sessions"],
            "best_epoch": r["best_epoch"], "ran_epochs": r["ran_epochs"],
            "secs": r["secs"], "n_train": r["n_train"], "n_val": r["n_val"],
            "accuracy": float(accuracy_score(r["true"], r["pred"])),
            "macro_f1": float(f1_score(r["true"], r["pred"], average="macro",
                                       zero_division=0)),
        } for r in results],
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {args.out}")


if __name__ == "__main__":
    main()
