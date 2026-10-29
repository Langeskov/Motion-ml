"""
Motion Encoder
==============
Transformer Encoder + AttentionPooling + ClassificationHead, 用于运动状态分类。

架构:
  Input(batch, seq_len=150, input_size=24)
    ↓ Linear Projection (input_size → d_model)
    ↓ Positional Encoding
    ↓ Transformer Encoder (nhead=4, layers=N)
    ↓ Attention Pooling → (batch, d_model)
    ↓ Classification Head (d_model → num_classes)
    ↓ CrossEntropy Loss

说明: 早期版本还有"自监督预训练 + 微调"路径 (MotionPretrainModel /
ReconstructionHead / MaskGenerator), 实测在本数据规模下不优于直接训练
(OOF macro F1 0.572 vs 0.785), 已移除。
"""

from typing import Optional

import torch
import torch.nn as nn

from models.common import PositionalEncoding, AttentionPooling  # noqa: F401


class ClassificationHead(nn.Module):
    """
    分类头: 将 Transformer 输出映射到类别概率。

    用于微调阶段。

    Args:
        d_model:     Transformer 特征维度
        num_classes: 输出类别数
        dropout:     Dropout 概率
    """

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, d_model]

        Returns:
            [batch, num_classes]
        """
        return self.head(x)


class MotionEncoder(nn.Module):
    """
    Transformer Encoder 主干网络。

    Args:
        input_size:  输入特征维度 (默认 21)
        d_model:     Transformer 特征维度 (默认 128)
        nhead:       Multi-head Attention 头数 (默认 4)
        num_layers:  Transformer Encoder 层数 (默认 2)
        dim_feedforward: FFN 隐藏层维度 (默认 d_model * 4)
        dropout:     Dropout 概率 (默认 0.1)
    """

    def __init__(
        self,
        input_size: int = 21,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        # 输入投影
        self.projection = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(
            d_model=d_model, max_len=500, dropout=dropout
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: [batch, seq_len, input_size]

        Returns:
            [batch, seq_len, d_model]
        """
        # 投影到 d_model 维度
        projected = self.projection(x)

        # 位置编码
        encoded = self.pos_encoding(projected)

        # Transformer Encoder
        output = self.transformer(encoded)

        return output

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MotionClassifier(nn.Module):
    """
    运动状态分类器: MotionEncoder + AttentionPooling + ClassificationHead。

    Args:
        input_size:  输入特征维度
        d_model:     Transformer 特征维度
        nhead:       Attention 头数
        num_layers:  Encoder 层数
        num_classes: 输出类别数
        dropout:     Dropout 概率
    """

    def __init__(
        self,
        input_size: int = 21,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.encoder = MotionEncoder(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.attn_pool = AttentionPooling(d_model)
        self.classifier = ClassificationHead(d_model, num_classes, dropout)

        # 缓存 attention weights (用于可视化)
        self._cached_attn_weights: Optional[torch.Tensor] = None
        # forward 里给 self 赋值张量属于副作用, 会让 ONNX 导出把 batch 维特化成具体值
        # (实测输出被钉成 [1, 5])。导出前把该开关置 False。
        self._capture_attention: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: [batch, seq_len, input_size]

        Returns:
            [batch, num_classes]  logits
        """
        # Encoder
        encoder_out = self.encoder(x)  # [batch, seq_len, d_model]

        # Attention Pooling
        pooled, attn_weights = self.attn_pool(encoder_out)
        if self._capture_attention:
            self._cached_attn_weights = attn_weights.detach()

        # Classification
        logits = self.classifier(pooled)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测类别 (带 softmax)。"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
        return probs

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """获取最近一次前向传播的 attention weights。"""
        return self._cached_attn_weights

    def count_parameters(self) -> int:
        """返回参数数量。"""
        return sum(p.numel() for p in self.parameters())


def build_classifier(
    input_size: int = 24,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    num_classes: int = 5,
    dropout: float = 0.3,
) -> MotionClassifier:
    """
    构建运动状态分类模型。

    Args:
        input_size:  输入特征维度
        d_model:     Transformer 特征维度
        nhead:       Attention 头数
        num_layers:  Encoder 层数
        num_classes: 输出类别数
        dropout:     Dropout 概率
    """
    model = MotionClassifier(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )

    print(f"[Model] MotionClassifier: {model.count_parameters():,} parameters")
    print(f"[Model] input_size={input_size}, d_model={d_model}, "
          f"nhead={nhead}, num_layers={num_layers}, num_classes={num_classes}")
    return model
