"""
MotionLSTM V2
=============
LSTM 模型，支持动态 input_size。

V2 变更:
  - input_size 由 dataset 自动传入 (默认 21)
  - 保持原有架构不变

架构:
    Input(21) → LSTM(128, 2层, dropout=0.3) → FC(128→5)

输入:  (batch, seq_len=150, input_size)
输出:  (batch, 5)  logits
"""

import torch
import torch.nn as nn


class MotionLSTM(nn.Module):
    """
    双层 LSTM 运动状态分类器。

    Args:
        input_size:  输入特征维度 (V2 默认 21)
        hidden_size: LSTM 隐藏层维度 (默认 128)
        num_layers:  LSTM 层数 (默认 2)
        output_size: 输出类别数 (默认 5)
        dropout:     Dropout 概率 (默认 0.3)
    """

    def __init__(
        self,
        input_size: int = 21,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 全连接分类头
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: [batch, seq_len, input_size]

        Returns:
            [batch, output_size]  logits
        """
        # LSTM 输出: [batch, seq_len, hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_size]

        # Dropout + 全连接
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)  # [batch, output_size]

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


def build_model(
    input_size: int = 21,
    hidden_size: int = 128,
    num_layers: int = 2,
    output_size: int = 5,
    dropout: float = 0.3,
) -> MotionLSTM:
    """
    构建 MotionLSTM 模型。

    Args:
        input_size: 输入特征维度 (V2 默认 21)

    Returns:
        MotionLSTM 实例
    """
    model = MotionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
    )
    print(f"[Model] MotionLSTM: {model.count_parameters():,} trainable parameters")
    print(f"[Model] input_size={input_size}, hidden_size={hidden_size}, "
          f"num_layers={num_layers}, output_size={output_size}")
    return model
