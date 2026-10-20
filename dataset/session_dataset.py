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
    fixed_label: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    滑动窗口切分。

    Args:
        features:     [N, D] 特征矩阵
        labels:       [N] 帧级标签
        window_size:  窗口大小
        stride:       滑动步长
        fixed_label:  >=0 时直接使用此标签 (Session 级)，否则用帧级多数投票

    Returns:
        (X, Y) 窗口特征和标签
    """
    X, Y = [], []
    n = len(features)

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        window_feat = features[start:end]
        if fixed_label >= 0:
            label = fixed_label
        else:
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
        drop_features: Optional[list[str]] = None,
        verbose: bool = True,
    ):
        if isinstance(sessions, SessionDataset):
            session_list = sessions.sessions
        else:
            session_list = sessions

        if builder is None:
            builder = FeatureBuilder()

        self.feature_map = builder.feature_map  # 暴露给下游
        self.drop_features = list(drop_features) if drop_features else []

        aligner = TimeAligner(max_gap_ms=max_gap_ms)
        extractor = FeatureExtractor(rolling_window=rolling_window)

        # ── 1. 逐 Session: 对齐 → 特征工程 → 标准化 → 滑动窗口 ──
        all_window_X = []
        all_window_Y = []
        total_matched = 0
        total_imu = 0
        total_windows = 0

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

            # 提取特征列 (可剔除与任务无关/依赖设备的特征, 如绝对姿态角)
            feature_cols = [
                c for c in extractor.ALL_COLS
                if c in clean_df.columns and c not in self.drop_features
            ]
            feat = clean_df[feature_cols].values.astype(np.float32)
            labels = clean_df["label"].map(LABEL_MAP).values

            # 过滤无效标签
            valid_mask = ~np.isnan(labels)
            feat = feat[valid_mask]
            labels = labels[valid_mask].astype(np.int64)

            # Step D: 逐 Session 做滑动窗口 (防止跨 Session 数据泄漏)
            session_label = LABEL_MAP.get(session.label.upper().strip(), -1)
            if session_label < 0:
                if verbose:
                    print(f"  [{session.session_id}] 跳过: 未知标签 '{session.label}'")
                continue

            session_X, session_Y = _sliding_windows(
                feat, labels, window_size, stride, fixed_label=session_label
            )
            n_windows = len(session_X)
            if n_windows == 0:
                if verbose:
                    print(f"  [{session.session_id}] 跳过: 帧数 {len(feat)} < 窗口大小 {window_size}")
                continue

            all_window_X.append(session_X)
            all_window_Y.append(session_Y)
            total_windows += n_windows

            if verbose:
                print(f"  [{session.session_id}] 窗口: {n_windows} 个")

        if not all_window_X:
            raise ValueError("没有可用的 Session 数据 (所有 Session 帧数不足或标签无效)")

        # ── 2. 拼接窗口 (不再跨 Session) ──────────────
        features_cat = np.concatenate(all_window_X, axis=0)  # [total_windows, window_size, D]
        labels_cat = np.concatenate(all_window_Y, axis=0)    # [total_windows]

        # ── 3. 标准化 (在窗口维度上展平→标准化→恢复形状) ──
        n_windows_total, ws, feat_dim = features_cat.shape
        flat_features = features_cat.reshape(-1, feat_dim)

        if scaler is None:
            scaler = StandardScaler()

        if fit_scaler:
            flat_features = scaler.fit_transform(flat_features).astype(np.float32)
        else:
            flat_features = scaler.transform(flat_features).astype(np.float32)

        self.scaler = scaler
        self.feature_dim = feat_dim

        # ── 4. 恢复窗口形状 ──────────────────────────
        self.X = flat_features.reshape(n_windows_total, ws, feat_dim)
        self.Y = labels_cat

        if verbose:
            print(f"[MotionWindowDataset] Sessions: {len(session_list)}, "
                  f"窗口: {total_windows}, 特征维度: {self.feature_dim}")
            print(f"  窗口大小: {window_size}, 步长: {stride}")
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
    repair_timestamps: bool = True,
    drop_features: Optional[list[str]] = None,
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
    ds = SessionDataset(data_dir, repair_timestamps=repair_timestamps)
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
        drop_features=drop_features,
    )

    return dataset, dataset.scaler


# ══════════════════════════════════════════════════════════════
# Session 级划分 (可信评测协议)
# ══════════════════════════════════════════════════════════════
#
# 为什么必须按 Session 划分:
#   滑动窗口 stride=50 < window=150, 相邻窗口共享 100/150 帧。
#   窗口级 random_split 会让验证窗口与训练窗口在同一段采集数据上重叠,
#   实测泄漏率 99.84%, 验证指标退化为"训练集精度"。
#
#   Session 级划分 = 留出整段采集, 验证集是模型从未见过的数据。


def build_per_session_windows(
    data_dir: str = "data/raw",
    window_size: int = 150,
    stride: int = 50,
    rolling_window: int = 10,
    max_gap_ms: float = 2000.0,
    builder: Optional[FeatureBuilder] = None,
    repair_timestamps: bool = True,
    drop_features: Optional[list[str]] = None,
) -> list[dict]:
    """
    逐 Session 构建窗口 (未标准化), 返回会话级窗口列表。

    用于 Session 级训练/验证划分: 标准化参数只能在训练会话上拟合。

    Returns:
        [{"session_id": str, "label": str, "label_id": int,
          "X": [n, window, D], "Y": [n], "n_frames": int}, ...]
    """
    ds = SessionDataset(data_dir, verbose=False,
                        repair_timestamps=repair_timestamps)
    out = []

    # 先探出特征维度, 再构造恒等 scaler (不改变数值, 只绕过未拟合检查)
    probe_session = ds.sessions[0]
    probe = MotionWindowDataset(
        sessions=[probe_session], window_size=window_size, stride=stride,
        rolling_window=rolling_window, max_gap_ms=max_gap_ms, builder=builder,
        drop_features=drop_features,
        verbose=False,
    )
    dim = probe.feature_dim
    ident = StandardScaler()
    ident.mean_ = np.zeros(dim)
    ident.scale_ = np.ones(dim)
    ident.var_ = np.ones(dim)
    ident.n_features_in_ = dim

    for s in ds.sessions:
        label_id = LABEL_MAP.get(s.label.upper().strip(), -1)
        if label_id < 0:
            print(f"  [跳过] {s.session_id}: 未知标签 '{s.label}'")
            continue
        d = MotionWindowDataset(
            sessions=[s], window_size=window_size, stride=stride,
            scaler=ident, fit_scaler=False,
            rolling_window=rolling_window, max_gap_ms=max_gap_ms, builder=builder,
            drop_features=drop_features,
            verbose=False,
        )
        if len(d.X) == 0:
            print(f"  [跳过] {s.session_id}: 帧数不足 {window_size}")
            continue
        out.append({
            "session_id": s.session_id,
            "label": s.label,
            "label_id": label_id,
            "X": d.X,
            "Y": d.Y,
            "n_frames": len(d.X),
        })

    if not out:
        raise ValueError("没有可用的 Session (全部帧数不足或标签无效)")
    return out


def session_level_split(
    per_session: list[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[str]]:
    """
    按 Session 分组划分训练/验证集, 每类至少留出 1 个会话。

    策略: 每类内按窗口数 **从小到大** 依次放入验证集, 直到该类的验证窗口
    占比达到 val_ratio。这样大会话(通常信息量更丰富)优先留给训练集,
    同时验证集比例接近目标。
    该类只有 1 个会话时无法留出, 全部进训练集。

    Returns:
        (train_parts, val_parts, val_session_ids)
    """
    rng = np.random.RandomState(seed)
    by_label: dict[int, list[dict]] = {}
    for p in per_session:
        by_label.setdefault(p["label_id"], []).append(p)

    train_parts, val_parts = [], []
    for label_id in sorted(by_label):
        group = by_label[label_id]

        if len(group) == 1:
            train_parts.extend(group)
            print(f"  [警告] 类别 {group[0]['label']} 只有 1 个会话, 无法留出验证集")
            continue

        # 从小到大; 同尺寸时用 seed 打散, 保证可复现且不总选同一个
        jitter = rng.rand(len(group))
        order = np.lexsort((jitter, [p["X"].shape[0] for p in group]))
        group = [group[i] for i in order]

        total_win = sum(p["X"].shape[0] for p in group)
        target = total_win * val_ratio

        # 在所有"由小到大累加"的前缀中, 选窗口数最接近目标的一个。
        # 不能简单地"加到超过目标就停": 若下一个会话很大, 会严重过冲
        # (实测 CAR 会从 19% 直接跳到 46%)。
        best_n, best_gap = 1, None
        acc = 0
        for n, p in enumerate(group[:-1], start=1):   # 至少留 1 个会话给训练
            acc += p["X"].shape[0]
            gap = abs(acc - target)
            if best_gap is None or gap < best_gap:
                best_gap, best_n = gap, n

        held = group[:best_n]
        held_ids = {p["session_id"] for p in held}
        val_parts.extend(held)
        train_parts.extend(p for p in group if p["session_id"] not in held_ids)

    val_ids = [p["session_id"] for p in val_parts]
    return train_parts, val_parts, val_ids


def fit_scaler_on_windows(X: np.ndarray) -> StandardScaler:
    """只在给定窗口上拟合标准化参数 (避免验证集信息泄漏)。"""
    sc = StandardScaler()
    sc.fit(X.reshape(-1, X.shape[-1]))
    return sc


def apply_scaler_to_windows(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """应用标准化, 保持 [n, window, D] 形状。"""
    shp = X.shape
    out = scaler.transform(X.reshape(-1, shp[-1]))
    return out.reshape(shp).astype(np.float32)


def concat_windows(parts: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """拼接会话窗口列表为 (X, Y)。"""
    X = np.concatenate([p["X"] for p in parts], axis=0)
    Y = np.concatenate([p["Y"] for p in parts], axis=0)
    return X, Y
