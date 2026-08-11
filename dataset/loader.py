"""
Motion Dataset Loader V2
========================
读取 data/raw 中的 CSV 文件，进行特征工程和预处理，生成滑动窗口数据集。

V2 变更:
  - 集成 FeatureExtractor，特征从 10 维扩展到 21 维
  - input_size 自动检测 (feature_dim)
  - 滚动统计窗口 10 帧 (0.2s @ 50Hz)
"""

import glob
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from dataset.preprocess import FeatureExtractor

# ── 类别映射 ──────────────────────────────────────────────
LABEL_MAP = {
    "STATIONARY": 0,
    "WALKING": 1,
    "CYCLING": 2,
    "CAR": 3,
    "TRAIN": 4,
}
NUM_CLASSES = len(LABEL_MAP)

# ── 特征列 (由 FeatureExtractor 定义) ────────────────────
FEATURE_COLS = FeatureExtractor.ALL_COLS  # 21 维
INPUT_SIZE = len(FEATURE_COLS)            # 21


def load_csv_files(data_dir: str) -> pd.DataFrame:
    """
    读取 data_dir 下所有 CSV 文件并合并为单个 DataFrame。

    Args:
        data_dir: CSV 文件目录路径

    Returns:
        合并后的 DataFrame，按 (文件序号, timestamp) 排序
    """
    pattern = os.path.join(data_dir, "*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["_source_file"] = os.path.basename(f)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"[Dataset] Loaded {len(files)} CSV files, {len(combined)} total rows")
    return combined


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗:
    1. 删除无效 GPS (lat=0 & lon=0)
    2. 缺失值前向填充 + 后向填充
    3. 删除仍残留 NaN 的行
    """
    initial_len = len(df)

    # 删除无效 GPS
    invalid_gps = (df["latitude"] == 0) & (df["longitude"] == 0)
    df = df[~invalid_gps].copy()

    # 只填充原始列 (FeatureExtractor 会生成派生列)
    raw_cols = [c for c in FeatureExtractor.RAW_COLS if c in df.columns]
    df[raw_cols] = df[raw_cols].ffill().bfill()

    # 删除残留 NaN
    df = df.dropna(subset=raw_cols).reset_index(drop=True)

    cleaned = initial_len - len(df)
    if cleaned > 0:
        print(f"[Dataset] Cleaned {cleaned} rows ({initial_len} -> {len(df)})")
    return df


def normalize_features(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True,
) -> tuple[np.ndarray, StandardScaler]:
    """
    标准化传感器特征 (包含派生特征)。

    Args:
        df: 包含 FEATURE_COLS 的 DataFrame
        scaler: 已有的 StandardScaler (fit=False 时使用)
        fit: 是否拟合 scaler

    Returns:
        (标准化后的特征矩阵 [N, 21], scaler)
    """
    features = df[FEATURE_COLS].values.astype(np.float32)

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)

    return features, scaler


def create_sliding_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int = 150,
    stride: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    滑动窗口切分。

    Args:
        features: [N, input_size] 传感器特征
        labels:   [N] 每帧标签
        window_size: 窗口大小 (默认 150)
        stride: 滑动步长 (默认 50)

    Returns:
        X: [num_samples, window_size, input_size]
        Y: [num_samples]
    """
    X, Y = [], []
    n = len(features)

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        window = features[start:end]
        # 窗口内取众数标签
        window_labels = labels[start:end]
        label = np.bincount(window_labels).argmax()

        X.append(window)
        Y.append(label)

    X = np.stack(X)
    Y = np.array(Y, dtype=np.int64)

    print(f"[Dataset] Created {len(X)} windows "
          f"(size={window_size}, stride={stride}, features={features.shape[1]})")
    return X, Y


class MotionDataset(Dataset):
    """PyTorch Dataset，用于训练和评估。"""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def build_dataset(
    data_dir: str = "data/raw",
    window_size: int = 150,
    stride: int = 50,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
    rolling_window: int = 10,
) -> tuple[MotionDataset, StandardScaler]:
    """
    一站式构建数据集 (V2)。

    流程: 读取 → 清洗 → 特征工程 → 标签编码 → 标准化 → 滑动窗口

    Args:
        data_dir: CSV 数据目录
        window_size: 滑动窗口大小
        stride: 滑动步长
        scaler: 已有的 StandardScaler
        fit_scaler: 是否拟合 scaler
        rolling_window: 特征提取滚动窗口大小

    Returns:
        (MotionDataset, fitted scaler)
    """
    # 1. 读取 CSV
    df = load_csv_files(data_dir)

    # 2. 清洗
    df = clean_data(df)

    # 3. 特征工程 (V2 新增)
    extractor = FeatureExtractor(rolling_window=rolling_window)
    df = extractor.transform(df)
    print(f"[Dataset] Feature engineering: {len(extractor.RAW_COLS)} raw "
          f"+ {len(extractor.DERIVED_COLS)} derived = {extractor.feature_dim} features")

    # 4. 标签编码
    df["label"] = df["label"].str.upper().str.strip()
    unknown = set(df["label"].unique()) - set(LABEL_MAP.keys())
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")

    labels = df["label"].map(LABEL_MAP).values

    # 5. 标准化
    features, scaler = normalize_features(df, scaler, fit_scaler)

    # 6. 滑动窗口
    X, Y = create_sliding_windows(features, labels, window_size, stride)

    # 7. 类别分布统计
    unique, counts = np.unique(Y, return_counts=True)
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    print("[Dataset] Class distribution:")
    for u, c in zip(unique, counts):
        print(f"  {inv_map[u]:>12s}: {c:6d} ({c/len(Y)*100:.1f}%)")

    return MotionDataset(X, Y), scaler
