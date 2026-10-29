"""
test_pipeline.py
================
管线完整性测试: 模型构建 / 前向反向 / 模型工厂 / ONNX 导出元数据。
"""

import os
import sys

import numpy as np
import torch

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INPUT_SIZE = 24   # FeatureExtractor 27 维去掉 yaw/pitch/roll 后的实际输入


def test_motion_encoder():
    """测试 MotionEncoder / MotionClassifier 前向。"""
    print("\n[TEST] MotionEncoder / MotionClassifier")
    from models.motion_encoder import MotionEncoder, MotionClassifier

    encoder = MotionEncoder(input_size=INPUT_SIZE, d_model=64, nhead=4, num_layers=2)
    x = torch.randn(2, 150, INPUT_SIZE)
    out = encoder(x)
    assert out.shape == (2, 150, 64), f"Encoder output: {out.shape}"
    print(f"  ✓ MotionEncoder: {x.shape} -> {out.shape}")

    classifier = MotionClassifier(
        input_size=INPUT_SIZE, d_model=64, nhead=4, num_layers=2, num_classes=5
    )
    logits = classifier(x)
    assert logits.shape == (2, 5), f"Logits: {logits.shape}"
    print(f"  ✓ MotionClassifier: logits {logits.shape}")

    # 注意力权重缓存
    attn = classifier.get_attention_weights()
    assert attn is not None and attn.shape[0] == 2, f"attn: {attn.shape}"
    print(f"  ✓ attention weights: {attn.shape}")


def test_classifier_forward_backward():
    """测试分类器前向+反向传播。"""
    print("\n[TEST] Classifier Forward/Backward")
    from models.motion_encoder import MotionClassifier

    model = MotionClassifier(
        input_size=INPUT_SIZE, d_model=64, nhead=4, num_layers=2, num_classes=5
    )
    x = torch.randn(4, 150, INPUT_SIZE)
    labels = torch.randint(0, 5, (4,))

    logits = model(x)
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Classifier loss: {loss.item():.4f}")


def test_other_models():
    """测试其余三种架构的前向形状。"""
    print("\n[TEST] Other architectures")
    from models import MODEL_TYPES, build_model_by_type

    x = torch.randn(2, 150, INPUT_SIZE)
    for mt in sorted(MODEL_TYPES):
        model = build_model_by_type(mt, input_size=INPUT_SIZE,
                                    hidden_size=64, num_layers=2, output_size=5)
        out = model(x)
        assert out.shape == (2, 5), f"{mt} output: {out.shape}"
        print(f"  ✓ {mt:<18s} -> {tuple(out.shape)}")


def test_model_factory():
    """测试模型工厂的默认值与异常。"""
    print("\n[TEST] Model Factory")
    from models import MODEL_TYPES, build_model_by_type

    assert "motion_encoder" in MODEL_TYPES

    model = build_model_by_type("motion_encoder")
    out = model(torch.randn(1, 150, INPUT_SIZE))
    assert out.shape == (1, 5)
    print(f"  ✓ 默认构建 motion_encoder -> {tuple(out.shape)}")

    try:
        build_model_by_type("nonexistent")
        raise AssertionError("应当抛出 ValueError")
    except ValueError as e:
        print(f"  ✓ 非法 model_type 抛出 ValueError: {str(e)[:50]}...")


def test_training_helpers():
    """测试 train.py 里的公共工具 (EMA / 学习率调度 / 数据增强)。"""
    print("\n[TEST] Training helpers")
    from train import ModelEMA, build_scheduler, augment_batch

    # EMA: decay=0 时立即等于权重本身
    m = torch.nn.Linear(4, 4)
    ema = ModelEMA(m, decay=0.0)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)
    assert torch.allclose(ema.state_dict()["weight"], m.state_dict()["weight"])
    print("  ✓ ModelEMA (decay=0 立即同步)")

    # 调度器: warmup 后衰减
    opt = torch.optim.Adam(m.parameters(), lr=1.0)
    sched = build_scheduler(opt, steps_per_epoch=10, total_epochs=10, mode="cosine")
    lrs = []
    for _ in range(100):
        lrs.append(sched.get_last_lr()[0])
        sched.step()
    assert lrs[0] < lrs[9], "warmup 阶段 lr 应递增"
    assert lrs[-1] < lrs[9], "末期 lr 应衰减"
    print(f"  ✓ build_scheduler: lr 从 {lrs[0]:.4f} 升到 {max(lrs):.4f} 再降到 {lrs[-1]:.5f}")

    # 数据增强: 形状不变、值有变化
    x = torch.randn(3, 150, INPUT_SIZE)
    y = augment_batch(x, jitter=0.1, scale_range=(0.9, 1.1), shift_max=5)
    assert y.shape == x.shape
    assert not torch.equal(x, y)
    print("  ✓ augment_batch: 形状不变, 内容已扰动")


def main():
    print("=" * 60)
    print("  Pipeline Verification")
    print("=" * 60)

    tests = [
        test_motion_encoder,
        test_classifier_forward_backward,
        test_other_models,
        test_model_factory,
        test_training_helpers,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
