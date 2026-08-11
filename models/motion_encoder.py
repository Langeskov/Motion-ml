"""
Motion Encoder V5
=================
自监督预训练的 Transformer Encoder。

支持两种模式:
  1. 预训练模式: Encoder + Reconstruction Head → 恢复被 mask 的传感器数据
  2. 微调模式:   Encoder + Classification Head → 运动状态分类

架构:
  Pretraining:
    Input(batch, seq_len, input_size)
      ↓ Masking
      ↓ Linear Projection (input_size → d_model)
      ↓ Positional Encoding
      ↓ Transformer Encoder (nhead=4, layers=N)
      ↓ Reconstruction Head (d_model → input_size)
      ↓ MSE Loss (仅计算 mask 位置)

  Fine-tuning:
    Input(batch, seq_len, input_size)
      ↓ Linear Projection (input_size → d_model)
      ↓ Positional Encoding
      ↓ Transformer Encoder (加载预训练权重)
      ↓ Attention Pooling
      ↓ Classification Head (d_model → num_classes)
      ↓ CrossEntropy Loss
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    正弦余弦位置编码。

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
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """
    注意力池化层。

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            pooled: [batch, d_model]
            weights: [batch, seq_len]
        """
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return pooled, weights


class ReconstructionHead(nn.Module):
    """
    重建头: 将 Transformer 输出映射回原始传感器维度。

    用于预训练阶段，预测被 mask 位置的传感器值。

    Args:
        d_model:     Transformer 特征维度
        input_size:  原始传感器特征维度
    """

    def __init__(self, d_model: int, input_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, input_size]
        """
        return self.head(x)


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
    运动编码器: Transformer Encoder 主干网络。

    可用于预训练 (配合 ReconstructionHead) 和微调 (配合 ClassificationHead)。

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


class MotionPretrainModel(nn.Module):
    """
    预训练模型: MotionEncoder + ReconstructionHead。

    用于 Masked Sensor Modeling 自监督预训练。

    Args:
        input_size:  输入特征维度
        d_model:     Transformer 特征维度
        nhead:       Attention 头数
        num_layers:  Encoder 层数
        dropout:     Dropout 概率
    """

    def __init__(
        self,
        input_size: int = 21,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.encoder = MotionEncoder(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.reconstruction_head = ReconstructionHead(d_model, input_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        Args:
            x:    [batch, seq_len, input_size]  被 mask 后的输入
            mask: [batch, seq_len]  bool 张量，None 表示全部计算

        Returns:
            reconstructed: [batch, seq_len, input_size]  重建输出
            encoder_out:   [batch, seq_len, d_model]      Encoder 输出
        """
        encoder_out = self.encoder(x)
        reconstructed = self.reconstruction_head(encoder_out)
        return reconstructed, encoder_out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        仅编码 (用于微调时提取特征)。

        Args:
            x: [batch, seq_len, input_size]

        Returns:
            [batch, seq_len, d_model]
        """
        return self.encoder(x)

    def get_encoder_state_dict(self) -> dict:
        """获取 encoder 的状态字典 (用于保存预训练权重)。"""
        return self.encoder.state_dict()

    def load_encoder_state_dict(self, state_dict: dict):
        """加载 encoder 的状态字典。"""
        self.encoder.load_state_dict(state_dict)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MotionClassifier(nn.Module):
    """
    微调分类模型: MotionEncoder + AttentionPooling + ClassificationHead。

    加载预训练的 encoder 权重，然后进行有监督微调。

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

        # 缓存 attention weights
        self._cached_attn_weights: Optional[torch.Tensor] = None

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

    def load_pretrained_encoder(self, state_dict: dict):
        """加载预训练的 encoder 权重。"""
        self.encoder.load_state_dict(state_dict)
        print(f"[Model] Loaded pretrained encoder weights")

    def freeze_encoder(self):
        """冻结 encoder 参数 (仅训练分类头)。"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(f"[Model] Encoder frozen")

    def unfreeze_encoder(self):
        """解冻 encoder 参数。"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print(f"[Model] Encoder unfrozen")

    def count_parameters(self) -> int:
        """返回总参数数量 (包括冻结的)。"""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """返回可训练参数数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_pretrain_model(
    input_size: int = 21,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> MotionPretrainModel:
    """构建预训练模型。"""
    model = MotionPretrainModel(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    )
    print(f"[Model] MotionPretrainModel: {model.count_parameters():,} parameters")
    print(f"[Model] input_size={input_size}, d_model={d_model}, "
          f"nhead={nhead}, num_layers={num_layers}")
    return model


def build_classifier(
    input_size: int = 21,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    num_classes: int = 5,
    dropout: float = 0.3,
    pretrained_encoder_path: str = None,
) -> MotionClassifier:
    """
    构建微调分类模型。

    Args:
        input_size:  输入特征维度
        d_model:     Transformer 特征维度
        nhead:       Attention 头数
        num_layers:  Encoder 层数
        num_classes: 输出类别数
        dropout:     Dropout 概率
        pretrained_encoder_path: 预训练 encoder 权重路径 (可选)
    """
    model = MotionClassifier(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    )

    if pretrained_encoder_path is not None:
        import torch
        ckpt = torch.load(pretrained_encoder_path, map_location="cpu", weights_only=False)
        encoder_state = ckpt.get("encoder_state_dict", ckpt.get("model_state_dict"))
        if encoder_state:
            # 提取 encoder 前缀
            encoder_keys = {k: v for k, v in encoder_state.items()
                           if k.startswith("encoder.")}
            if encoder_keys:
                # 去掉 "encoder." 前缀
                clean_keys = {k.replace("encoder.", ""): v for k, v in encoder_keys.items()}
                model.load_pretrained_encoder(clean_keys)
            else:
                model.load_pretrained_encoder(encoder_state)

    print(f"[Model] MotionClassifier: {model.count_parameters():,} parameters")
    print(f"[Model] input_size={input_size}, d_model={d_model}, "
          f"nhead={nhead}, num_layers={num_layers}, num_classes={num_classes}")
    return model
