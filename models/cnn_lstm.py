"""
CNN-LSTM V3
===========
CNN-LSTM 混合模型，用于运动状态分类。

核心思想:
  IMU 数据同时具有:
  1. 高频局部震动模式 → 1D CNN 擅长捕捉
  2. 长时间运动趋势   → LSTM 擅长捕捉

架构:
  Input(batch, seq_len, input_size)
    ↓ Transpose → (batch, input_size, seq_len)
    ↓ 1D CNN Feature Extractor
    ↓   Conv1D(in→64, k=5) → BN → ReLU → MaxPool(2)
    ↓   Conv1D(64→128, k=3) → BN → ReLU → MaxPool(2)
    ↓ Transpose back → (batch, seq_len', 128)
    ↓ LSTM Temporal Encoder (hidden=128, layers=2)
    ↓ Linear Classifier → 5 classes

输入:  (batch, seq_len=150, input_size)
输出:  (batch, 5)  logits
"""

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    1D CNN 特征提取器。

    从传感器时间序列中提取局部模式特征。
    两层卷积 + BatchNorm + ReLU + MaxPool。

    Args:
        input_size: 输入特征维度 (默认 21)
        cnn_channels: CNN 输出通道数 (默认 128)
    """

    def __init__(self, input_size: int = 21, cnn_channels: int = 128):
        super().__init__()

        self.conv_block = nn.Sequential(
            # 第一层: 大卷积核捕捉局部震动模式
            nn.Conv1d(
                in_channels=input_size,
                out_channels=64,
                kernel_size=5,
                padding=2,  # 保持序列长度
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 长度减半

            # 第二层: 小卷积核提取更抽象的特征
            nn.Conv1d(
                in_channels=64,
                out_channels=cnn_channels,
                kernel_size=3,
                padding=1,  # 保持序列长度
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 长度再减半
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len]  (已转置)

        Returns:
            [batch, cnn_channels, seq_len']
        """
        return self.conv_block(x)


class CNNLSTM(nn.Module):
    """
    CNN-LSTM 混合运动状态分类器。

    流程:
      1. CNN 提取局部高频特征 (震动、冲击)
      2. LSTM 建模长时间运动趋势 (加速、转弯)
      3. FC 分类

    Args:
        input_size:   输入特征维度 (默认 21)
        cnn_channels: CNN 输出通道数 (默认 128)
        hidden_size:  LSTM 隐藏层维度 (默认 128)
        num_layers:   LSTM 层数 (默认 2)
        output_size:  输出类别数 (默认 5)
        dropout:      Dropout 概率 (默认 0.3)
    """

    def __init__(
        self,
        input_size: int = 21,
        cnn_channels: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_size = input_size
        self.cnn_channels = cnn_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ── CNN 特征提取器 ──
        self.cnn = CNNFeatureExtractor(
            input_size=input_size,
            cnn_channels=cnn_channels,
        )

        # ── LSTM 时序编码器 ──
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── 分类头 ──
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

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

        # 2. CNN 特征提取: [batch, cnn_channels, seq_len']
        cnn_features = self.cnn(x)

        # 3. 转置回来: [batch, seq_len', cnn_channels]
        cnn_features = cnn_features.transpose(1, 2)

        # 4. LSTM 时序编码: [batch, seq_len', hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)

        # 5. 取最后时间步: [batch, hidden_size]
        last_hidden = lstm_out[:, -1, :]

        # 6. 分类: [batch, output_size]
        dropped = self.dropout(last_hidden)
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

    def count_parameters(self) -> int:
        """返回可训练参数数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_cnn_lstm(
    input_size: int = 21,
    cnn_channels: int = 128,
    hidden_size: int = 128,
    num_layers: int = 2,
    output_size: int = 5,
    dropout: float = 0.3,
) -> CNNLSTM:
    """
    构建 CNN-LSTM 混合模型。

    Returns:
        CNNLSTM 实例
    """
    model = CNNLSTM(
        input_size=input_size,
        cnn_channels=cnn_channels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
    )
    print(f"[Model] CNNLSTM: {model.count_parameters():,} trainable parameters")
    print(f"[Model] input_size={input_size}, cnn_channels={cnn_channels}, "
          f"hidden_size={hidden_size}, num_layers={num_layers}")
    return model
