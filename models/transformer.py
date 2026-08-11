"""
CNN-Transformer V4
==================
CNN-Transformer 混合模型，用于运动状态分类。

核心思想:
  1. CNN 提取局部高频特征 (震动、冲击)
  2. Transformer Encoder 建模长距离时序依赖 (高铁启动、汽车加速)
  3. Attention Pooling 聚合序列表示
  4. 支持保存 attention weights 用于可视化

架构:
  Input(batch, seq_len=150, input_size=21)
    ↓ Transpose → (batch, input_size, seq_len)
    ↓ CNN Feature Extractor
    ↓   Conv1D(in→64, k=5) → BN → ReLU → MaxPool(2)
    ↓   Conv1D(64→128, k=3) → BN → ReLU → MaxPool(2)
    ↓ Transpose back → (batch, seq_len', 128)
    ↓ Positional Encoding
    ↓ Transformer Encoder (nhead=4, layers=2)
    ↓ Attention Pooling → (batch, 128)
    ↓ Linear Classifier → 5 classes
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    正弦余弦位置编码。

    为 Transformer 输入添加时序位置信息。
    使用 sin/cos 函数生成，无需学习参数。

    Args:
        d_model: 特征维度
        max_len: 最大序列长度
        dropout: Dropout 概率
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """
    注意力池化层。

    通过学习注意力权重，对序列进行加权聚合。
    相比取最后时间步或平均池化，能更好地聚焦于重要时序位置。

    Args:
        d_model: 特征维度
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            pooled: [batch, d_model]
            weights: [batch, seq_len]  注意力权重 (可用于可视化)
        """
        # 计算每个时间步的注意力分数
        scores = self.attention(x).squeeze(-1)  # [batch, seq_len]
        weights = torch.softmax(scores, dim=-1)  # [batch, seq_len]

        # 加权聚合
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [batch, d_model]

        return pooled, weights


class CNNTransformer(nn.Module):
    """
    CNN-Transformer 混合运动状态分类器。

    流程:
      1. CNN 提取局部高频特征 (震动、冲击)
      2. Transformer Encoder 建模长距离依赖
      3. Attention Pooling 聚合序列表示
      4. FC 分类

    Args:
        input_size:    输入特征维度 (默认 21)
        cnn_channels:  CNN 输出通道数 (默认 128)
        d_model:       Transformer 特征维度 (默认 128)
        nhead:         Multi-head Attention 头数 (默认 4)
        num_layers:    Transformer Encoder 层数 (默认 2)
        output_size:   输出类别数 (默认 5)
        dropout:       Dropout 概率 (默认 0.3)
    """

    def __init__(
        self,
        input_size: int = 21,
        cnn_channels: int = 128,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        output_size: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_size = input_size
        self.cnn_channels = cnn_channels
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_size = output_size

        # ── CNN 特征提取器 ──
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # ── 投影层: CNN 输出维度 → d_model ──
        self.projection = nn.Linear(cnn_channels, d_model)

        # ── 位置编码 ──
        self.pos_encoding = PositionalEncoding(
            d_model=d_model, max_len=500, dropout=dropout
        )

        # ── Transformer Encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ── Attention Pooling ──
        self.attn_pool = AttentionPooling(d_model)

        # ── 分类头 ──
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)

        # ── 缓存 attention weights (用于可视化) ──
        self._cached_attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: [batch, seq_len, input_size]

        Returns:
            [batch, output_size]  logits
        """
        # 1. 转置: [batch, seq_len, input_size] → [batch, input_size, seq_len]
        x = x.transpose(1, 2)

        # 2. CNN: [batch, cnn_channels, seq_len']
        cnn_features = self.cnn(x)

        # 3. 转置回来: [batch, seq_len', cnn_channels]
        cnn_features = cnn_features.transpose(1, 2)

        # 4. 投影到 d_model 维度
        projected = self.projection(cnn_features)  # [batch, seq_len', d_model]

        # 5. 位置编码
        encoded = self.pos_encoding(projected)

        # 6. Transformer Encoder: [batch, seq_len', d_model]
        transformer_out = self.transformer(encoded)

        # 7. Attention Pooling: [batch, d_model], [batch, seq_len']
        pooled, attn_weights = self.attn_pool(transformer_out)

        # 缓存 attention weights (用于可视化)
        self._cached_attn_weights = attn_weights.detach()

        # 8. 分类
        dropped = self.dropout(pooled)
        logits = self.fc(dropped)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别 (带 softmax)。

        Args:
            x: [batch, seq_len, input_size]

        Returns:
            [batch, output_size]  概率分布
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
        return probs

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        获取最近一次前向传播的 attention weights。

        Returns:
            [batch, seq_len'] 或 None (如果还没有前向传播)
        """
        return self._cached_attn_weights

    def forward_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播并返回 attention weights。

        Args:
            x: [batch, seq_len, input_size]

        Returns:
            logits: [batch, output_size]
            attn_weights: [batch, seq_len']
        """
        logits = self.forward(x)
        attn_weights = self._cached_attn_weights
        return logits, attn_weights

    def count_parameters(self) -> int:
        """返回可训练参数数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_cnn_transformer(
    input_size: int = 21,
    cnn_channels: int = 128,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    output_size: int = 5,
    dropout: float = 0.3,
) -> CNNTransformer:
    """
    构建 CNN-Transformer 混合模型。

    Returns:
        CNNTransformer 实例
    """
    model = CNNTransformer(
        input_size=input_size,
        cnn_channels=cnn_channels,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
    )
    print(f"[Model] CNNTransformer: {model.count_parameters():,} trainable parameters")
    print(f"[Model] input_size={input_size}, cnn_channels={cnn_channels}, "
          f"d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
    return model
