# motion-ml models package
"""
模型工厂模块 V5。

支持通过 model_type 字符串选择模型:
  - "lstm"            → MotionLSTM (纯 LSTM)
  - "cnn_lstm"        → CNNLSTM (CNN-LSTM 混合)
  - "cnn_transformer" → CNNTransformer (CNN-Transformer 混合)
  - "motion_encoder"  → MotionEncoder (自监督预训练 Transformer)

V5 新增:
  - MotionEncoder: 自监督预训练的 Transformer Encoder
  - MotionPretrainModel: 预训练模型 (Encoder + Reconstruction Head)
  - MotionClassifier: 微调分类模型 (Encoder + AttentionPooling + ClassificationHead)
"""

import torch.nn as nn

from models.motion_lstm import MotionLSTM, build_model as build_lstm
from models.cnn_lstm import CNNLSTM, build_cnn_lstm
from models.transformer import CNNTransformer, build_cnn_transformer
from models.motion_encoder import (
    MotionEncoder,
    MotionPretrainModel,
    MotionClassifier,
    build_pretrain_model,
    build_classifier,
)


MODEL_TYPES = {"lstm", "cnn_lstm", "cnn_transformer", "motion_encoder"}


def build_model_by_type(
    model_type: str = "lstm",
    input_size: int = 21,
    hidden_size: int = 128,
    num_layers: int = 2,
    output_size: int = 5,
    dropout: float = 0.3,
) -> nn.Module:
    """
    根据 model_type 构建模型。

    Args:
        model_type: "lstm", "cnn_lstm", "cnn_transformer" 或 "motion_encoder"
        input_size: 输入特征维度
        hidden_size: 隐藏层维度
        num_layers: 层数
        output_size: 输出类别数
        dropout: Dropout 概率

    Returns:
        模型实例
    """
    if model_type == "lstm":
        return build_lstm(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
        )
    elif model_type == "cnn_lstm":
        return build_cnn_lstm(
            input_size=input_size,
            cnn_channels=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
        )
    elif model_type == "cnn_transformer":
        return build_cnn_transformer(
            input_size=input_size,
            cnn_channels=hidden_size,
            d_model=hidden_size,
            nhead=4,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
        )
    elif model_type == "motion_encoder":
        return build_classifier(
            input_size=input_size,
            d_model=hidden_size,
            nhead=4,
            num_layers=num_layers,
            num_classes=output_size,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Supported: {sorted(MODEL_TYPES)}"
        )


def load_model_from_checkpoint(path: str, device=None) -> nn.Module:
    """
    从 checkpoint 加载模型 (自动识别 model_type)。

    Args:
        path: checkpoint 文件路径
        device: 目标设备

    Returns:
        加载了权重的模型实例
    """
    import torch

    if device is None:
        device = torch.device("cpu")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})

    model_type = args.get("model_type", "lstm")
    input_size = ckpt.get("input_size", args.get("input_size", 21))
    hidden_size = args.get("hidden_size", 128)
    num_layers = args.get("num_layers", 2)
    output_size = args.get("output_size", 5)
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

    return model
