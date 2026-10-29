"""
export.py
=========
导出训练好的模型, 供 PyTorch 与移动端 (ONNX) 使用。

用法:
    python export.py --checkpoint checkpoints/best_motion_encoder.pth --onnx

产出:
    exported/motion_<model_type>.pth   纯权重 + 元数据 (含 scaler 参数)
    exported/motion_<model_type>.onnx  单文件 ONNX, batch 维动态

导出后请用 verify_onnx.py 验证, 导出成功不代表结果正确。
"""

import argparse
import os

import numpy as np
import torch

from dataset.session_dataset import LABEL_MAP
from models import build_model_by_type


def parse_args():
    parser = argparse.ArgumentParser(description="Export Motion Model V5")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_lstm.pth")
    parser.add_argument("--output_dir", type=str, default="exported")
    parser.add_argument("--onnx", action="store_true", help="Export to ONNX format")
    return parser.parse_args()


def load_model(path: str, device: torch.device):
    """加载 checkpoint 并返回模型。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})

    model_type = ckpt.get("model_type", args.get("model_type", "motion_encoder"))
    input_size = ckpt.get("input_size", args.get("input_size", 24))

    model = build_model_by_type(
        model_type=model_type,
        input_size=input_size,
        hidden_size=args.get("hidden_size", 64),
        num_layers=args.get("num_layers", 2),
        output_size=len(LABEL_MAP),
        dropout=0.0,  # 导出时关闭 dropout
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt, model_type


def export_pth(model, checkpoint: dict, output_dir: str, model_type: str):
    """
    导出为 .pth 格式 (纯权重 + 元数据)。
    """
    path = os.path.join(output_dir, f"motion_{model_type}.pth")

    export_data = {
        "model_state_dict": model.state_dict(),
        "model_type": model_type,
        "input_size": model.input_size,
        "hidden_size": model.d_model if hasattr(model, 'd_model') else model.hidden_size,
        "num_layers": model.num_layers,
        "output_size": len(LABEL_MAP),
        "label_map": LABEL_MAP,
        "scaler_mean": checkpoint.get("scaler_mean"),
        "scaler_scale": checkpoint.get("scaler_scale"),
    }

    torch.save(export_data, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[Export] Saved .pth: {path} ({size_mb:.2f} MB)")
    return path


def export_onnx(model, output_dir: str, model_type: str, seq_len: int = 150):
    """
    导出为 ONNX 格式, batch 维设为动态 (移动端连续推理需要)。

    注意: torch 2.x 默认走 dynamo 导出器, 但 dynamo 遇到 nn.TransformerEncoder
    会把 batch 维特化成具体值 (无论 dynamic_shapes 怎么写, 输入输出都被钉成 [1, ...],
    batch>1 直接报错)。这里强制 dynamo=False 走 TorchScript 导出器,
    它能保住动态 batch。
    """
    try:
        path = os.path.join(output_dir, f"motion_{model_type}.onnx")

        dummy_input = torch.randn(1, seq_len, model.input_size)

        # 关闭 forward 里的注意力缓存副作用 (避免导出时给 self 赋值张量)
        had_capture = hasattr(model, "_capture_attention")
        prev = getattr(model, "_capture_attention", None)
        if had_capture:
            model._capture_attention = False

        try:
            # dynamo=False: 必须走 TorchScript 导出器。
            # dynamo 导出器遇到 nn.TransformerEncoder 会把 batch 维特化成具体值
            # (无论 dynamic_shapes 怎么写, 输入输出都被钉成 [1, ...], batch>1 直接报错),
            # 只有遗留导出器能保住动态 batch —— 移动端连续推理必需。
            torch.onnx.export(
                model,
                (dummy_input,),
                path,
                export_params=True,
                external_data=False,  # 权重内嵌, 产出单文件 .onnx
                opset_version=18,
                do_constant_folding=True,
                dynamo=False,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )
        finally:
            if had_capture:
                model._capture_attention = prev

        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[Export] Saved ONNX: {path} ({size_mb:.2f} MB)")
        return path

    except ImportError as e:
        # torch.onnx.export (dynamo) 依赖 onnx + onnxscript，
        # 缺少任何一个都会抛 ImportError，需指明真实缺失的包
        print(f"[Export] ONNX export failed — missing dependency: {e}")
        print("[Export] Try: pip install onnx onnxscript")
        return None
    except Exception as e:
        print(f"[Export] ONNX export failed: {e}")
        print("[Export] Skipping ONNX export.")
        return None


def main():
    args = parse_args()
    device = torch.device("cpu")
    print(f"[Export] Using {device}")

    # ── 1. 加载模型 ──────────────────────────────────
    print(f"[Export] Loading checkpoint: {args.checkpoint}")
    model, checkpoint, model_type = load_model(args.checkpoint, device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 2. 导出 .pth ─────────────────────────────────
    pth_path = export_pth(model, checkpoint, args.output_dir, model_type)

    # ── 3. 导出 ONNX (可选) ──────────────────────────
    if args.onnx:
        export_onnx(model, args.output_dir, model_type)
    else:
        print("[Export] ONNX skipped (use --onnx to enable)")

    # ── 4. 模型信息 ──────────────────────────────────
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n[Export] Model summary:")
    print(f"  Type:        {model_type}")
    print(f"  Parameters:  {param_count:,}")
    print(f"  Input shape: (batch, 150, {model.input_size})")
    print(f"  Output shape: (batch, {len(LABEL_MAP)})")
    print(f"  Classes:     {LABEL_MAP}")

    print("\n[Export] Done!")


if __name__ == "__main__":
    main()
