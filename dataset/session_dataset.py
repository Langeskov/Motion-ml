"""
Session-based PyTorch Dataset
==============================

将 Session 数据转换为可训练的滑动窗口 Dataset。
IMU 为主序列，GNSS 通过 TimeAligner 按时间戳对齐。

数据流:
  Session → TimeAligner → FeatureBuilder → FeatureExtractor → 标准化 → 滑动窗口 → Dataset
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from dataset.preprocess import FeatureExtractor
from dataset.session_loader import Session, SessionDataset
from dataset.time_aligner import TimeAligner
from dataset.feature_builder import FeatureBuilder, FeatureMap


# ── 类别映射 ──────────────────────────────────────────────
LABEL_MAP = {
    "STATIONARY": 0,
    "WALKING": 1,
    "CYCLING": 2,
    "CAR": 3,
    "TRAIN": 4,
}
NUM_CLASSES = len(LABEL_MAP)


def _sliding_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """滑动窗口切分。"""
    X, Y = [], []
    n = len(features)

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        window_feat = features[start:end]
        window_label = labels[start:end]
        label = np.bincount(window_label).argmax()
        X.append(window_feat)
        Y.append(label)

    if not X:
        return np.empty((0, window_size, features.shape[1]), dtype=np.float32), \
               np.empty(0, dtype=np.int64)

    return np.stack(X).astype(np.float32), np.array(Y, dtype=np.int64)


class MotionWindowDataset(Dataset):
    """
    基于滑动窗口的运动数据集。

    数据流 (每个 Session):
      IMU + GNSS
        → TimeAligner.align()     # gps_speed m/s + gps_valid + gps_dt_ms
        → FeatureBuilder.build()  # m/s→km/h, NaN→0, 特征矩阵 [N, D]
        → FeatureExtractor        # 派生特征 (21维)
        → StandardScaler          # 标准化
        → Sliding Windows         # (N, 150, 21)

    模型可通过 self.feature_map.index_of("gps_speed") 获取列索引，
    不依赖固定列号。

    Args:
        sessions:       SessionDataset 或 Session 列表
        window_size:    窗口大小 (帧数)
        stride:         滑动步长
        scaler:         已有的 StandardScaler (None 则新建)
        fit_scaler:     是否拟合 scaler
        rolling_window: 特征提取滚动窗口大小
        max_gap_ms:     TimeAligner 最大 GNSS 时间差 (毫秒)
        builder:        自定义 FeatureBuilder (None 则使用默认 10 维)
        verbose:        打印日志
    """

    def __init__(
        self,
        sessions,
        window_size: int = 150,
        stride: int = 50,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True,
        rolling_window: int = 10,
        max_gap_ms: float = 2000.0,
        builder: Optional[FeatureBuilder] = None,
        verbose: bool = True,
    ):
        if isinstance(sessions, SessionDataset):
            session_list = sessions.sessions
        else:
            session_list = sessions

        if builder is None:
            builder = FeatureBuilder()

        self.feature_map = builder.feature_map  # 暴露给下游

        aligner = TimeAligner(max_gap_ms=max_gap_ms)
        extractor = FeatureExtractor(rolling_window=rolling_window)

        # ── 1. 逐 Session: 对齐 → FeatureBuilder → FeatureExtractor ──
        all_features = []
        all_labels = []
        total_matched = 0
        total_imu = 0

        for session in session_list:
            # Step A: TimeAligner — GNSS m/s 对齐到 IMU 时间轴
            aligned, align_stats = aligner.align(session.imu, session.gnss)
            total_matched += align_stats.gps_matched
            total_imu += align_stats.total_imu

            if verbose:
                print(f"  [{session.session_id}] {align_stats}")

            # Step B: FeatureBuilder — 按 FeatureSpec 做单位转换+NaN处理
            base_features, fmap, build_stats = builder.build(aligned)

            if verbose:
                print(f"  [{session.session_id}] {build_stats}")

            # 重建 DataFrame: 使用 feature_map.columns 命名列 (不依赖固定顺序)
            clean_df = pd.DataFrame(base_features, columns=fmap.columns)
            clean_df["gps_valid"] = aligned["gps_valid"].values if "gps_valid" in aligned.columns else True

            # 确保 label 列存在
            if "label" not in aligned.columns:
                clean_df["label"] = session.label
            else:
                clean_df["label"] = aligned["label"].values

            clean_df["label"] = clean_df["label"].str.upper().str.strip()

            # Step C: FeatureExtractor — 基础特征 → 派生特征
            clean_df = extractor.transform(clean_df)

            # 提取特征列
            feature_cols = [c for c in extractor.ALL_COLS if c in clean_df.columns]
            feat = clean_df[feature_cols].values.astype(np.float32)
            labels = clean_df["label"].map(LABEL_MAP).values

            # 过滤无效标签
            valid_mask = ~np.isnan(labels)
            feat = feat[valid_mask]
            labels = labels[valid_mask].astype(np.int64)

            all_features.append(feat)
            all_labels.append(labels)

        if not all_features:
            raise ValueError("没有可用的 Session 数据")

        # ── 2. 拼接 ───────────────────────────────────
        features_cat = np.concatenate(all_features, axis=0)
        labels_cat = np.concatenate(all_labels, axis=0)

        # ── 3. 标准化 ───────────────────────────────────
        if scaler is None:
            scaler = StandardScaler()

        if fit_scaler:
            self.features = scaler.fit_transform(features_cat).astype(np.float32)
        else:
            self.features = scaler.transform(features_cat).astype(np.float32)

        self.scaler = scaler
        self.feature_dim = self.features.shape[1]

        # ── 4. 滑动窗口 ────────────────────────────────
        self.X, self.Y = _sliding_windows(
            self.features, labels_cat, window_size, stride
        )

        if verbose:
            print(f"[MotionWindowDataset] Sessions: {len(session_list)}, "
                  f"总帧数: {len(features_cat)}, 特征维度: {self.feature_dim}")
            print(f"  窗口: {window_size}, 步长: {stride}, "
                  f"样本数: {len(self.X)}")
            if total_imu > 0:
                print(f"  GNSS 对齐: {total_matched}/{total_imu} "
                      f"({total_matched / total_imu:.1%})")

            # 类别分布
            inv_map = {v: k for k, v in LABEL_MAP.items()}
            unique, counts = np.unique(self.Y, return_counts=True)
            print("  类别分布:")
            for u, c in zip(unique, counts):
                print(f"    {inv_map.get(u, '?'):>12s}: {c:6d} ({c / len(self.Y) * 100:.1f}%)")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def build_session_dataset(
    data_dir: str = "data/raw",
    window_size: int = 150,
    stride: int = 50,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
    rolling_window: int = 10,
    max_gap_ms: float = 2000.0,
    builder: Optional[FeatureBuilder] = None,
) -> tuple[MotionWindowDataset, StandardScaler]:
    """
    一站式构建数据集。

    流程: 扫描 → 加载 Session → TimeAligner → FeatureBuilder → FeatureExtractor
          → 标准化 → 滑动窗口

    Args:
        data_dir:       数据目录
        window_size:    窗口大小
        stride:         步长
        scaler:         已有 scaler
        fit_scaler:     是否拟合
        rolling_window: 特征滚动窗口
        max_gap_ms:     GNSS 最大时间差
        builder:        自定义 FeatureBuilder (None=默认10维)

    Returns:
        (MotionWindowDataset, fitted scaler)
    """
    ds = SessionDataset(data_dir)
    print(ds.summary())

    dataset = MotionWindowDataset(
        sessions=ds,
        window_size=window_size,
        stride=stride,
        scaler=scaler,
        fit_scaler=fit_scaler,
        rolling_window=rolling_window,
        max_gap_ms=max_gap_ms,
        builder=builder,
    )

    return dataset, dataset.scaler
