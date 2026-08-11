"""
FeatureExtractor
================
V2 传感器特征工程模块。

从原始 10 维传感器数据中提取 21 维特征:
  原始 (10): gps_speed, acc_x/y/z, gyro_x/y/z, roll, pitch, yaw
  新增 (11): acc_magnitude, gyro_magnitude, speed_delta,
             acc_rolling_mean, acc_rolling_std,
             gyro_rolling_mean, gyro_rolling_std,
             roll_rolling_mean, roll_rolling_std,
             pitch_rolling_mean, vibration_score

滚动统计窗口: 10 帧 (0.2s @ 50Hz)
"""

import numpy as np
import pandas as pd


class FeatureExtractor:
    """
    传感器特征提取器。

    将原始 10 维传感器列扩展为 21 维，保留原始列并新增派生列。
    所有滚动统计默认使用 10 帧窗口 (0.2 秒 @ 50Hz)。
    """

    # 原始传感器列
    RAW_COLS = [
        "gps_speed",
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "roll", "pitch", "yaw",
    ]

    # 新增特征列
    DERIVED_COLS = [
        "acc_magnitude",
        "gyro_magnitude",
        "speed_delta",
        "acc_rolling_mean",
        "acc_rolling_std",
        "gyro_rolling_mean",
        "gyro_rolling_std",
        "roll_rolling_mean",
        "roll_rolling_std",
        "pitch_rolling_mean",
        "vibration_score",
    ]

    # 全部特征列 (原始 + 新增)
    ALL_COLS = RAW_COLS + DERIVED_COLS

    def __init__(self, rolling_window: int = 10):
        """
        Args:
            rolling_window: 滚动统计窗口大小 (帧数)
        """
        self.rolling_window = rolling_window

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对 DataFrame 进行特征工程，原地新增派生列。

        Args:
            df: 包含 RAW_COLS 的 DataFrame (会被修改)

        Returns:
            添加了 DERIVED_COLS 的 DataFrame
        """
        w = self.rolling_window

        # ── 1. 加速度模长 ─────────────────────────────
        # 三轴加速度的欧几里得范数，反映总加速度强度
        df["acc_magnitude"] = np.sqrt(
            df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
        )

        # ── 2. 陀螺仪模长 ─────────────────────────────
        # 三轴角速度的欧几里得范数，反映总旋转强度
        df["gyro_magnitude"] = np.sqrt(
            df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2
        )

        # ── 3. 速度变化率 ─────────────────────────────
        # 相邻帧速度差，反映加减速状态
        df["speed_delta"] = df["gps_speed"].diff().fillna(0.0)

        # ── 4. 加速度滚动统计 ─────────────────────────
        # acc_magnitude 的均值和标准差，反映运动稳定性
        acc_roll = df["acc_magnitude"].rolling(window=w, min_periods=1)
        df["acc_rolling_mean"] = acc_roll.mean()
        df["acc_rolling_std"] = acc_roll.std().fillna(0.0)

        # ── 5. 陀螺仪滚动统计 ─────────────────────────
        # gyro_magnitude 的均值和标准差，反映旋转稳定性
        gyro_roll = df["gyro_magnitude"].rolling(window=w, min_periods=1)
        df["gyro_rolling_mean"] = gyro_roll.mean()
        df["gyro_rolling_std"] = gyro_roll.std().fillna(0.0)

        # ── 6. Roll/Pitch 滚动统计 ────────────────────
        # 姿态角的滚动均值，反映倾斜趋势
        df["roll_rolling_mean"] = (
            df["roll"].rolling(window=w, min_periods=1).mean()
        )
        df["roll_rolling_std"] = (
            df["roll"].rolling(window=w, min_periods=1).std().fillna(0.0)
        )
        df["pitch_rolling_mean"] = (
            df["pitch"].rolling(window=w, min_periods=1).mean()
        )

        # ── 7. 震动强度指标 ───────────────────────────
        # acc_magnitude 滚动标准差 / 滚动均值 (变异系数)
        # 静止时 ≈ 0，颠簸时数值大
        mean = df["acc_rolling_mean"].replace(0, np.nan)
        df["vibration_score"] = (df["acc_rolling_std"] / mean).fillna(0.0)

        return df

    @property
    def feature_dim(self) -> int:
        """返回总特征维度。"""
        return len(self.ALL_COLS)
