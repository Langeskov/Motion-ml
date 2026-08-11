"""
TimeAligner — GNSS ↔ IMU 时间对齐模块
========================================

将低频 GNSS (1~5Hz) 对齐到高频 IMU (100~200Hz) 时间轴上。

对齐原则:
  - 以 IMU 时间轴为基准，不删除任何 IMU 行
  - 每条 IMU 寻找时间最近的 GNSS 记录
  - 时间差超过 2 秒 → gps_speed=NaN, gps_valid=False
  - 不插值、不平滑，保留原始精度

输出列 (追加到 IMU DataFrame):
  - gps_speed:   对齐后的 GNSS speed (float, 可能 NaN)
  - gps_accuracy: 对齐后的 GNSS accuracy
  - gps_valid:    布尔标记，True 表示 GNSS 数据有效
  - gps_dt_ms:    与最近 GNSS 的时间差 (毫秒，调试用)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AlignStats:
    """对齐统计信息。"""
    total_imu: int = 0
    gps_matched: int = 0
    gps_out_of_range: int = 0
    gps_no_data: int = 0
    max_dt_ms: float = 0.0
    median_dt_ms: float = 0.0

    @property
    def match_rate(self) -> float:
        if self.total_imu == 0:
            return 0.0
        return self.gps_matched / self.total_imu

    def __repr__(self) -> str:
        return (
            f"AlignStats(imu={self.total_imu}, matched={self.gps_matched}, "
            f"oob={self.gps_out_of_range}, no_data={self.gps_no_data}, "
            f"match_rate={self.match_rate:.1%}, "
            f"max_dt={self.max_dt_ms:.0f}ms, median_dt={self.median_dt_ms:.0f}ms)"
        )


class TimeAligner:
    """
    GNSS → IMU 时间对齐器。

    Args:
        max_gap_ms: 最大允许时间差 (毫秒)，超过则标记为无效。
                    默认 2000ms (2 秒)。
    """

    def __init__(self, max_gap_ms: float = 2000.0):
        self.max_gap_ms = max_gap_ms

    def align(
        self,
        imu: pd.DataFrame,
        gnss: pd.DataFrame,
    ) -> tuple[pd.DataFrame, AlignStats]:
        """
        将 GNSS 数据对齐到 IMU 时间轴。

        Args:
            imu:  IMU DataFrame，必须含 'timestamp' 列
            gnss: GNSS DataFrame，必须含 'timestamp' 列。
                  可为空 DataFrame (无 GNSS 数据时)。

        Returns:
            (aligned_imu, stats)
            aligned_imu: 原始 IMU 列 + gps_speed / gps_accuracy / gps_valid / gps_dt_ms
            stats: 对齐统计
        """
        stats = AlignStats(total_imu=len(imu))

        # ── 情况 1: 无 GNSS 数据 ──────────────────────
        if gnss.empty or "timestamp" not in gnss.columns:
            imu = imu.copy()
            imu["gps_speed"] = np.nan
            imu["gps_accuracy"] = np.nan
            imu["gps_valid"] = False
            imu["gps_dt_ms"] = np.nan
            stats.gps_no_data = stats.total_imu
            return imu, stats

        # ── 情况 2: 有 GNSS 数据，执行对齐 ─────────────
        imu_sorted = imu.sort_values("timestamp").reset_index(drop=True)
        gnss_sorted = gnss.sort_values("timestamp").reset_index(drop=True)

        imu_ts = imu_sorted["timestamp"].values  # shape: [N]
        gnss_ts = gnss_sorted["timestamp"].values  # shape: [M]

        # 对每条 IMU，用 searchsorted 找最近 GNSS
        # idx = searchsorted(gnss_ts, imu_ts) → 插入位置
        idx = np.searchsorted(gnss_ts, imu_ts, side="left")

        # 候选: 左邻 (idx-1) 和右邻 (idx)，取时间差更小的
        best_gnss_idx = np.empty(len(imu_ts), dtype=np.int64)
        best_dt = np.empty(len(imu_ts), dtype=np.float64)

        for i in range(len(imu_ts)):
            candidates = []
            if idx[i] > 0:
                candidates.append(idx[i] - 1)
            if idx[i] < len(gnss_ts):
                candidates.append(idx[i])

            # 选时间差最小的
            best_c = min(candidates, key=lambda c: abs(imu_ts[i] - gnss_ts[c]))
            best_gnss_idx[i] = best_c
            best_dt[i] = abs(imu_ts[i] - gnss_ts[best_c])

        # 转换为毫秒
        dt_ms = best_dt.astype(np.float64)

        # 构建对齐列
        matched_mask = dt_ms <= self.max_gap_ms

        gps_speed = np.full(len(imu_ts), np.nan, dtype=np.float64)
        gps_accuracy = np.full(len(imu_ts), np.nan, dtype=np.float64)

        if "speed" in gnss_sorted.columns:
            gps_speed[matched_mask] = gnss_sorted["speed"].values[best_gnss_idx[matched_mask]]
        if "accuracy" in gnss_sorted.columns:
            gps_accuracy[matched_mask] = gnss_sorted["accuracy"].values[best_gnss_idx[matched_mask]]

        # 写入 DataFrame
        result = imu_sorted.copy()
        result["gps_speed"] = gps_speed
        result["gps_accuracy"] = gps_accuracy
        result["gps_valid"] = matched_mask
        result["gps_dt_ms"] = dt_ms

        # ── 统计 ─────────────────────────────────────
        stats.gps_matched = int(matched_mask.sum())
        stats.gps_out_of_range = int((~matched_mask).sum())
        if stats.gps_matched > 0:
            valid_dt = dt_ms[matched_mask]
            stats.max_dt_ms = float(np.max(valid_dt))
            stats.median_dt_ms = float(np.median(valid_dt))

        return result, stats
