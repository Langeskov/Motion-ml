"""
test_v5_pipeline.py
===================
验证 V5 自监督预训练管线的完整性。
生成合成数据进行端到端测试。
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pandas as pd
import torch

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_synthetic_csv(output_dir: str, num_files: int = 2, num_rows: int = 1000):
    """生成合成 CSV 数据用于测试。"""
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_files):
        data = {
            "timestamp": np.arange(num_rows) * 20,  # 50Hz
            "acc_x": np.random.randn(num_rows) * 2,
            "acc_y": np.random.randn(num_rows) * 2,
            "acc_z": np.random.randn(num_rows) * 2 + 9.8,
            "gyro_x": np.random.randn(num_rows) * 0.5,
            "gyro_y": np.random.randn(num_rows) * 0.5,
            "gyro_z": np.random.randn(num_rows) * 0.5,
            "mag_x": np.random.randn(num_rows) * 10,
            "mag_y": np.random.randn(num_rows) * 10,
            "mag_z": np.random.randn(num_rows) * 10,
            "roll": np.random.randn(num_rows) * 10,
            "pitch": np.random.randn(num_rows) * 10,
            "yaw": np.random.randn(num_rows) * 180,
            "latitude": np.random.uniform(30, 40, num_rows),
            "longitude": np.random.uniform(100, 120, num_rows),
            "gps_speed": np.random.uniform(0, 30, num_rows),  # km/h
            "label": np.random.choice(["STATIONARY", "WALKING", "CYCLING", "CAR", "TRAIN"], num_rows),
        }
        df = pd.DataFrame(data)
        path = os.path.join(output_dir, f"test_data_{i}.csv")
        df.to_csv(path, index=False)
        print(f"  Generated: {path} ({num_rows} rows)")


def test_mask_generator():
    """测试 MaskGenerator。"""
    print("\n[TEST] MaskGenerator")
    from models.mask_generator import MaskGenerator

    gen = MaskGenerator(mask_ratio=0.15, mask_strategy="random")
    x = torch.randn(4, 150, 21)

    masked_x, mask = gen(x)
    assert masked_x.shape == x.shape, f"Shape mismatch: {masked_x.shape}"
    assert mask.shape == (4, 150), f"Mask shape: {mask.shape}"
    assert mask.dtype == torch.bool

    # 被 mask 的位置应该是 0
    for b in range(4):
        for t in range(150):
            if mask[b, t]:
                assert (masked_x[b, t] == 0).all(), "Masked position not zero"

    print("  ✓ MaskGenerator OK")


def test_motion_encoder():
    """测试 MotionEncoder 模型。"""
    print("\n[TEST] MotionEncoder")
    from models.motion_encoder import MotionEncoder, MotionPretrainModel, MotionClassifier

    # 测试 Encoder
    encoder = MotionEncoder(input_size=21, d_model=64, nhead=4, num_layers=2)
    x = torch.randn(2, 150, 21)
    out = encoder(x)
    assert out.shape == (2, 150, 64), f"Encoder output: {out.shape}"
    print(f"  ✓ MotionEncoder: {x.shape} -> {out.shape}")

    # 测试 PretrainModel
    pretrain = MotionPretrainModel(input_size=21, d_model=64, nhead=4, num_layers=2)
    mask = torch.zeros(2, 150, dtype=torch.bool)
    mask[:, 50:80] = True
    reconstructed, enc_out = pretrain(x, mask)
    assert reconstructed.shape == x.shape, f"Reconstructed: {reconstructed.shape}"
    assert enc_out.shape == (2, 150, 64), f"Enc out: {enc_out.shape}"
    print(f"  ✓ MotionPretrainModel: reconstructed {reconstructed.shape}")

    # 测试 Classifier
    classifier = MotionClassifier(input_size=21, d_model=64, nhead=4, num_layers=2, num_classes=5)
    logits = classifier(x)
    assert logits.shape == (2, 5), f"Logits: {logits.shape}"
    print(f"  ✓ MotionClassifier: logits {logits.shape}")


def test_pretrain_forward_backward():
    """测试预训练前向+反向传播。"""
    print("\n[TEST] Pretrain Forward/Backward")
    from models.motion_encoder import MotionPretrainModel
    from models.mask_generator import MaskGenerator

    model = MotionPretrainModel(input_size=21, d_model=64, nhead=4, num_layers=2)
    mask_gen = MaskGenerator(mask_ratio=0.15)

    x = torch.randn(4, 150, 21)
    masked_x, mask = mask_gen(x)

    reconstructed, _ = model(masked_x, mask)

    # 计算 MSE loss (仅 mask 位置)
    mask_expanded = mask.unsqueeze(-1).float()
    loss = ((reconstructed - x) ** 2 * mask_expanded).sum() / mask_expanded.sum()

    loss.backward()
    print(f"  ✓ Pretrain loss: {loss.item():.6f}")


def test_classifier_forward_backward():
    """测试分类器前向+反向传播。"""
    print("\n[TEST] Classifier Forward/Backward")
    from models.motion_encoder import MotionClassifier

    model = MotionClassifier(input_size=21, d_model=64, nhead=4, num_layers=2, num_classes=5)
    x = torch.randn(4, 150, 21)
    labels = torch.randint(0, 5, (4,))

    logits = model(x)
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    print(f"  ✓ Classifier loss: {loss.item():.4f}")


def test_pretrain_to_finetune():
    """测试预训练权重加载到微调模型。"""
    print("\n[TEST] Pretrain -> Finetune Weight Transfer")
    from models.motion_encoder import MotionPretrainModel, MotionClassifier

    # 预训练
    pretrain = MotionPretrainModel(input_size=21, d_model=64, nhead=4, num_layers=2)
    enc_state = pretrain.get_encoder_state_dict()

    # 微调
    classifier = MotionClassifier(input_size=21, d_model=64, nhead=4, num_layers=2, num_classes=5)
    classifier.load_pretrained_encoder(enc_state)

    # 验证权重一致
    for key in enc_state:
        assert torch.equal(
            classifier.encoder.state_dict()[key], enc_state[key]
        ), f"Weight mismatch: {key}"

    print("  ✓ Weight transfer OK")


def test_freeze_unfreeze():
    """测试冻结/解冻。"""
    print("\n[TEST] Freeze/Unfreeze")
    from models.motion_encoder import MotionClassifier

    model = MotionClassifier(input_size=21, d_model=64, nhead=4, num_layers=2, num_classes=5)

    # 冻结
    model.freeze_encoder()
    for param in model.encoder.parameters():
        assert not param.requires_grad, "Encoder not frozen"

    # 解冻
    model.unfreeze_encoder()
    for param in model.encoder.parameters():
        assert param.requires_grad, "Encoder still frozen"

    print("  ✓ Freeze/Unfreeze OK")


def test_dataset_loader():
    """测试数据加载管线。"""
    print("\n[TEST] Dataset Loader")

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_dir = os.path.join(tmpdir, "csv")
        generate_synthetic_csv(csv_dir, num_files=2, num_rows=500)

        from dataset.loader import load_csv_files, clean_data, normalize_features, create_sliding_windows
        from dataset.preprocess import FeatureExtractor

        df = load_csv_files(csv_dir)
        assert len(df) == 1000, f"Rows: {len(df)}"

        df = clean_data(df)

        # 需要先运行 FeatureExtractor 生成派生特征
        extractor = FeatureExtractor()
        df = extractor.transform(df)

        features, scaler = normalize_features(df)
        assert features.shape[0] > 0

        labels = np.zeros(len(features), dtype=np.int64)
        X, Y = create_sliding_windows(features, labels, window_size=150, stride=50)
        assert X.shape[1] == 150
        assert X.shape[0] > 0
        print(f"  ✓ Dataset: X={X.shape}, Y={Y.shape}")


def test_build_functions():
    """测试 build_* 函数。"""
    print("\n[TEST] Build Functions")
    from models.motion_encoder import build_pretrain_model, build_classifier

    pretrain = build_pretrain_model(input_size=21, d_model=64, nhead=4, num_layers=2)
    assert pretrain is not None
    print(f"  ✓ build_pretrain_model: {pretrain.count_parameters():,} params")

    classifier = build_classifier(input_size=21, d_model=64, nhead=4, num_layers=2, num_classes=5)
    assert classifier is not None
    print(f"  ✓ build_classifier: {classifier.count_parameters():,} params")


def test_model_factory():
    """测试模型工厂。"""
    print("\n[TEST] Model Factory")
    from models import build_model_by_type, MODEL_TYPES

    assert "motion_encoder" in MODEL_TYPES
    model = build_model_by_type("motion_encoder", input_size=21, hidden_size=64, num_layers=2, output_size=5)
    x = torch.randn(1, 150, 21)
    out = model(x)
    assert out.shape == (1, 5)
    print(f"  ✓ Factory: motion_encoder -> {out.shape}")


def main():
    print("=" * 60)
    print("  V5 Pipeline Verification")
    print("=" * 60)

    tests = [
        test_mask_generator,
        test_motion_encoder,
        test_pretrain_forward_backward,
        test_classifier_forward_backward,
        test_pretrain_to_finetune,
        test_freeze_unfreeze,
        test_dataset_loader,
        test_build_functions,
        test_model_factory,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
