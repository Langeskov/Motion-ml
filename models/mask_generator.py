"""
Mask Generator
==============
为 Masked Sensor Modeling 生成随机掩码。

策略:
  - 随机选择时间步进行 mask
  - 支持按比例 (mask_ratio) 控制 mask 数量
  - 支持连续 mask (block mask) 和随机 mask (random mask)
  - mask 位置用 0 填充，未 mask 位置保持原始值

输入:  (batch, seq_len, input_size)
输出:  masked_x, mask
  - masked_x: [batch, seq_len, input_size]  被 mask 后的输入
  - mask:     [batch, seq_len]  bool 张量，True 表示被 mask 的位置
"""

import torch


def generate_random_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float = 0.15,
    device: torch.device = None,
) -> torch.Tensor:
    """
    生成随机掩码。

    每个样本独立随机选择 mask_ratio 比例的时间步。

    Args:
        batch_size: 批次大小
        seq_len:    序列长度
        mask_ratio: mask 比例 (0.0 ~ 1.0)
        device:     目标设备

    Returns:
        mask: [batch, seq_len] bool 张量，True 表示被 mask
    """
    num_masked = max(1, int(seq_len * mask_ratio))

    # 为每个样本生成随机排列，取前 num_masked 个作为 mask 位置
    rand = torch.rand(batch_size, seq_len, device=device)
    _, indices = rand.topk(num_masked, dim=1)

    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    mask.scatter_(1, indices, True)

    return mask


def generate_block_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float = 0.15,
    block_size: int = 10,
    device: torch.device = None,
) -> torch.Tensor:
    """
    生成连续块状掩码。

    每个样本随机选择一个起始位置，mask 连续 block_size 个时间步。
    更适合捕捉连续运动片段的缺失。

    Args:
        batch_size: 批次大小
        seq_len:    序列长度
        mask_ratio: mask 比例 (用于计算 block 数量)
        block_size: 每个 block 的长度
        device:     目标设备

    Returns:
        mask: [batch, seq_len] bool 张量
    """
    total_masked = max(1, int(seq_len * mask_ratio))
    num_blocks = max(1, total_masked // block_size)

    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    for b in range(batch_size):
        # 随机选择 block 起始位置
        starts = torch.randint(0, max(1, seq_len - block_size + 1), (num_blocks,))
        for s in starts:
            end = min(s + block_size, seq_len)
            mask[b, s:end] = True

    return mask


def apply_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
    mask_value: float = 0.0,
) -> torch.Tensor:
    """
    将 mask 应用到输入序列。

    被 mask 的位置替换为 mask_value，未 mask 的位置保持原始值。

    Args:
        x:          [batch, seq_len, input_size]  原始输入
        mask:       [batch, seq_len]  bool 张量
        mask_value: mask 位置的填充值

    Returns:
        masked_x: [batch, seq_len, input_size]
    """
    # 扩展 mask 维度以匹配 input_size
    # mask: [batch, seq_len] → [batch, seq_len, 1]
    mask_expanded = mask.unsqueeze(-1)

    # 被 mask 的位置替换为 mask_value
    masked_x = torch.where(mask_expanded, torch.full_like(x, mask_value), x)

    return masked_x


class MaskGenerator:
    """
    掩码生成器，支持多种 mask 策略。

    Args:
        mask_ratio:   mask 比例 (默认 0.15)
        block_size:   block mask 的块大小 (默认 10)
        mask_strategy: "random" 或 "block"
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        block_size: int = 10,
        mask_strategy: str = "random",
    ):
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        self.mask_strategy = mask_strategy

    def generate(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        生成掩码。

        Args:
            batch_size: 批次大小
            seq_len:    序列长度
            device:     目标设备

        Returns:
            mask: [batch, seq_len] bool 张量
        """
        if self.mask_strategy == "block":
            return generate_block_mask(
                batch_size, seq_len, self.mask_ratio, self.block_size, device
            )
        else:
            return generate_random_mask(
                batch_size, seq_len, self.mask_ratio, device
            )

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对输入序列进行 mask。

        Args:
            x:    [batch, seq_len, input_size]
            mask: 可选的预生成 mask，如果为 None 则自动生成

        Returns:
            masked_x: [batch, seq_len, input_size]
            mask:     [batch, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        if mask is None:
            mask = self.generate(batch_size, seq_len, x.device)

        masked_x = apply_mask(x, mask)
        return masked_x, mask
