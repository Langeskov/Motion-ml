"""
FeatureExtractor
================
V2 传感器特征工程模块。

从原始 10 维传感器数据中提取 27 维特征:
  原始 (10): gps_speed, acc_x/y/z, gyro_x/y/z, roll, pitch, yaw
  时域派生 (11): acc_magnitude, gyro_magnitude, speed_delta,
             acc_rolling_mean, acc_rolling_std,
             gyro_rolling_mean, gyro_rolling_std,
             roll_rolling_mean, roll_rolling_std,
             pitch_rolling_mean, vibration_score
  频域派生 (6): acc_bp_0.5_2, acc_bp_2_5, acc_bp_5_10, acc_bp_10_25,
             acc_centroid, acc_dom_freq

滚动统计窗口: 10 帧 (0.2s @ 50Hz)
频域分析窗口: 100 帧 (2.0s @ 50Hz), 逐 10 帧算一次后插值回逐帧

为什么要有频域特征:
  电动车的振动集中在 2~5 Hz (电机/踩踏), 汽车能量更偏向 10~25 Hz 的宽带噪声。
  实测两类在 2-5 Hz 与 10-25 Hz 能量占比上分开得很干净
  (0.56 vs 0.24 / 0.08 vs 0.35), 而纯时域统计捕捉不到这个差异。
"""

import numpy as np
import pandas as pd

# 频带边界 (Hz), 与诊断实验保持一致
BAND_COLS = ["acc_bp_0.5_2", "acc_bp_2_5", "acc_bp_5_10", "acc_bp_10_25"]
BAND_EDGES = [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 25.0)]

SPECTRAL_COLS = BAND_COLS + ["acc_centroid", "acc_dom_freq"]


class FeatureExtractor:
    """
    传感器特征提取器。

    将原始 10 维传感器列扩展为 27 维，保留原始列并新增派生列。
    时域滚动统计默认使用 10 帧窗口 (0.2 秒 @ 50Hz)。

    Args:
        rolling_window:  时域滚动统计窗口 (帧数)
        fs:              采样率 (Hz)。频域特征需要它把 FFT 频点换算成 Hz,
                         必须与数据实际采样率一致 (重采样后即 --target_hz)。
        spectral_window: 频域分析窗口 (帧数)。100 帧 @ 50Hz = 2 秒,
                         频率分辨率 0.5 Hz, 足以区分 0.5-2 / 2-5 等频带。
        spectral_stride: 频域计算步长 (帧数)。逐 10 帧算一次再插值回逐帧,
                         避免对每帧都做 FFT。
    """

    # 原始传感器列
    RAW_COLS = [
        "gps_speed",
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z",
        "roll", "pitch", "yaw",
    ]

    # 新增特征列 (时域 + 频域)
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
    ] + SPECTRAL_COLS

    # 全部特征列 (原始 + 新增)
    ALL_COLS = RAW_COLS + DERIVED_COLS

    def __init__(
        self,
        rolling_window: int = 10,
        fs: float = 50.0,
        spectral_window: int = 100,
        spectral_stride: int = 10,
    ):
        if fs <= 0:
            raise ValueError(f"fs 必须 > 0, 收到 {fs}")
        self.rolling_window = rolling_window
        self.fs = float(fs)
        self.spectral_window = int(spectral_window)
        self.spectral_stride = max(1, int(spectral_stride))

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

        # ── 8. 频域特征 ───────────────────────────────
        # 在 acc_magnitude 上做滑窗 FFT，捕捉电动车(2~5Hz)与汽车(10~25Hz)的能量差异
        self._add_spectral(df)

        return df

    def _add_spectral(self, df: pd.DataFrame) -> None:
        """
        在 df 中新增 SPECTRAL_COLS (逐帧)。

        做法: 对 acc_magnitude 每 spectral_stride 帧做一次 spectral_window 帧的 FFT,
        再线性插值回逐帧。这样既保留逐帧形状 (窗口按帧切), 又不必逐帧做 FFT。
        """
        signal = df["acc_magnitude"].to_numpy(dtype=np.float64)
        n = len(signal)
        w, step = self.spectral_window, self.spectral_stride

        if n < w:
            # 样本太少, 不足以做一次完整 FFT
            for col in SPECTRAL_COLS:
                df[col] = 0.0
            return

        starts = np.arange(0, n - w + 1, step, dtype=np.int64)
        if starts[-1] != n - w:
            starts = np.append(starts, n - w)          # 保证最后一帧也被覆盖

        # 构造滑窗矩阵 [n_win, w]
        idx = starts[:, None] + np.arange(w, dtype=np.int64)[None, :]
        windows = signal[idx]

        # 去均值 + Hann 窗, 抑制频谱泄漏
        windows = windows - windows.mean(axis=1, keepdims=True)
        windows = windows * np.hanning(w)[None, :]

        spec = np.abs(np.fft.rfft(windows, axis=1)) ** 2     # [n_win, n_bins]
        freqs = np.fft.rfftfreq(w, d=1.0 / self.fs)          # [n_bins]
        total = spec.sum(axis=1, keepdims=True) + 1e-12      # [n_win, 1]

        frame_idx = np.arange(n, dtype=np.float64)

        # 频带能量占比
        for col, (lo, hi) in zip(BAND_COLS, BAND_EDGES):
            mask = (freqs >= lo) & (freqs < hi)
            if not mask.any():
                vals = np.zeros(len(starts))
            else:
                vals = spec[:, mask].sum(axis=1) / total[:, 0]
            df[col] = np.interp(frame_idx, starts.astype(np.float64), vals)

        # 谱质心 (Hz): 频率的加权平均
        centroid = (spec * freqs[None, :]).sum(axis=1) / total[:, 0]
        df["acc_centroid"] = np.interp(
            frame_idx, starts.astype(np.float64), centroid
        )

        # 主频 (Hz): 能量最大的频点
        dom = freqs[np.argmax(spec, axis=1)]
        df["acc_dom_freq"] = np.interp(
            frame_idx, starts.astype(np.float64), dom
        )

    @property
    def feature_dim(self) -> int:
        """返回总特征维度。"""
        return len(self.ALL_COLS)
