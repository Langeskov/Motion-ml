"""验证导出的 ONNX 模型: 形状 / 动态 batch / 与 PyTorch 的数值一致性 / 精度保持。"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.session_dataset import (
    LABEL_MAP, apply_scaler_to_windows, build_per_session_windows,
    concat_windows, fit_scaler_on_windows,
)
from models import build_model_by_type

ONNX_PATH = "exported/motion_motion_encoder.onnx"
CKPT = "checkpoints/best_motion_encoder.pth"

print("=" * 64)
print("  ONNX 导出验证")
print("=" * 64)

# ── 1. 基本结构 ──
import onnx

m = onnx.load(ONNX_PATH)
onnx.checker.check_model(m)
print(f"[1] onnx.checker 通过  |  opset = "
      f"{m.opset_import[0].version}  |  文件 = {os.path.getsize(ONNX_PATH)/1024:.0f} KB")
print(f"    输入 : {[(i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]}")
print(f"    输出 : {[(o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]}")

# ── 2. 加载 PyTorch 原模型 ──
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
args = ck["args"]
pt = build_model_by_type(
    model_type=ck["model_type"], input_size=ck["input_size"],
    hidden_size=args.get("hidden_size", 64), num_layers=args.get("num_layers", 2),
    output_size=len(LABEL_MAP), dropout=0.0,
)
pt.load_state_dict(ck["model_state_dict"])
pt.eval()

# ── 3. 数值一致性 (多种 batch size, 验证动态轴) ──
import onnxruntime as ort

sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
iname = sess.get_inputs()[0].name

print(f"\n[2] PyTorch vs ONNX 数值一致性:")
rng = np.random.default_rng(0)
worst = 0.0
for bs in (1, 2, 3, 8, 32):
    x = rng.standard_normal((bs, 150, 24)).astype(np.float32)
    with torch.no_grad():
        ref = pt(torch.from_numpy(x)).numpy()
    got = sess.run(None, {iname: x})[0]
    diff = float(np.abs(ref - got).max())
    worst = max(worst, diff)
    ok = "OK" if diff < 1e-4 else "偏差偏大"
    print(f"    batch={bs:<3d} shape={got.shape}  max|Δ|={diff:.2e}  {ok}")
print(f"    最大偏差: {worst:.2e}")

# ── 4. 真实数据精度 (与训练时 evaluate 一致) ──
print(f"\n[3] 真实数据端到端精度:")
per = build_per_session_windows("data/raw", 150, 50,
                                drop_features=["yaw", "pitch", "roll"])
X_raw, Y = concat_windows(per)
scaler = fit_scaler_on_windows(X_raw)
X = apply_scaler_to_windows(scaler, X_raw)
del X_raw

pt_preds, on_preds = [], []
for i in range(0, len(X), 256):
    xb = torch.from_numpy(X[i:i + 256])
    with torch.no_grad():
        pt_preds.append(pt(xb).argmax(1).numpy())
    on_preds.append(sess.run(None, {iname: X[i:i + 256].astype(np.float32)})[0].argmax(1))
pt_preds = np.concatenate(pt_preds)
on_preds = np.concatenate(on_preds)

agree = (pt_preds == on_preds).mean()
pt_acc = (pt_preds == Y).mean()
on_acc = (on_preds == Y).mean()
print(f"    样本数: {len(Y):,}")
print(f"    PyTorch/ONNX 预测一致率: {agree:.6%}   (应为 100%)")
print(f"    PyTorch accuracy: {pt_acc:.4f}")
print(f"    ONNX    accuracy: {on_acc:.4f}")
print(f"    差异: {abs(pt_acc - on_acc):.2e}   (应为 0)")

# ── 5. 结论 ──
print(f"\n{'=' * 64}")
ok = worst < 1e-4 and agree > 0.9999
print(f"  {'✅ ONNX 导出验证通过' if ok else '❌ 验证未通过'}")
print(f"{'=' * 64}")
sys.exit(0 if ok else 1)
