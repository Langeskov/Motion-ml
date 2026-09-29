"""
models/common.py
================
公共模型组件，供多个模型架构复用。

包含:
  - PositionalEncoding: 正弦余弦位置编码
  - AttentionPooling:   注意力池化层
"""

import math

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
        scores = self.attention(x).squeeze(-1)  # [batch, seq_len]
        weights = torch.softmax(scores, dim=-1)  # [batch, seq_len]
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [batch, d_model]
        return pooled, weights