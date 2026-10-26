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


def _sliding_windows_by_segment(
    features: np.ndarray,
    labels: np.ndarray,
    segments,
    window_size: int,
    stride: int,
    fixed_label: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    只在同一个连续段内切窗口, 避免一个窗口跨越录音中断 (实测最长停顿 525 秒)。

    segments 为每行的段号; 为 None 时退化为普通滑动窗口。
    """
    if segments is None or len(segments) != len(features):
        return _sliding_windows(features, labels, window_size, stride, fixed_label)

    change = np.where(np.diff(segments) != 0)[0] + 1
    bounds = np.concatenate([[0], change, [len(segments)]])

    Xs, Ys = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < window_size:
            continue
        xs, ys = _sliding_windows(
            features[a:b], labels[a:b], window_size, stride, fixed_label
        )
        if len(xs):
            Xs.append(xs)
            Ys.append(ys)

    if not Xs:
        return (np.empty((0, window_size, features.shape[1]), dtype=np.float32),
                np.empty(0, dtype=np.int64))
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


# ── 重采样 ──────────────────────────────────────────────────
#
# 为什么必须重采样:
#   各会话的 IMU 有效采样率差异极大 (实测 4.9 ~ 250 Hz)。窗口长度以"帧数"定义,
#   所以同样 150 帧, 在 250 Hz 会话上是 0.6 秒, 在 50 Hz 会话上是 3 秒。
#   滚动统计 (10 帧窗口) 的物理含义也随之改变 —— 模型被迫在两种时间尺度间折中。
#
#   统一重采样到固定频率后, "150 帧" 才有确定含义, 跨会话特征才可比。


def estimate_rate_hz(timestamps, max_gap_ms: float = 1000.0) -> float:
    """
    估计采集的**瞬时采样率** (Hz)。

    用活跃间隔的**中位数**而不是均值: 采集常出现密集段 + 次秒级停顿,
    均值会被停顿拉低 (实测同一会话均值 13.7 Hz, 中位数 90.9 Hz),
    而决定"能否降采样/是否会上采样"的是采样点本身的间隔。

    Args:
        timestamps: 时间戳序列 (毫秒)
        max_gap_ms: 超过此间隔视为停顿, 不计入

    Returns:
        采样率 (Hz); 无法估计时返回 0.0
    """
    t = np.sort(np.asarray(timestamps, dtype=np.float64))
    if len(t) < 2:
        return 0.0
    d = np.diff(t)
    active = d[(d > 0) & (d < max_gap_ms)]
    if len(active) == 0:
        return 0.0
    return float(1000.0 / np.median(active))


def assign_segments(timestamps, max_gap_ms: float = 1000.0) -> np.ndarray:
    """
    按时间间隙把样本切成连续段, 返回每行的段号 (从 0 开始)。

    两个作用:
      1. 重采样只在段内进行, 不会把停顿"补"成插值数据
      2. 滑动窗口只在段内切, 避免一个窗口跨越大段录音中断
    """
    t = np.asarray(timestamps, dtype=np.float64)
    seg = np.zeros(len(t), dtype=np.int32)
    if len(t) > 1:
        seg[1:] = np.cumsum(np.diff(t) > max_gap_ms)
    return seg


def _resample_uniform(sub: pd.DataFrame, target_hz: float, time_col: str) -> pd.DataFrame:
    """把一段连续数据重采样到固定频率 (数值列插值, 类别/布尔列最近邻)。"""
    t = sub[time_col].to_numpy(dtype=np.float64)
    dur_s = (t[-1] - t[0]) / 1000.0
    n = int(np.floor(dur_s * target_hz)) + 1
    if n < 2:
        return sub

    grid = t[0] + np.arange(n) * (1000.0 / target_hz)
    nn_idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 1)

    out = {time_col: grid}
    for col in sub.columns:
        if col == time_col:
            continue
        s = sub[col]
        vals = s.to_numpy()
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            out[col] = np.interp(grid, t, vals.astype(np.float64))
        else:
            out[col] = vals[nn_idx]
    return pd.DataFrame(out)


def resample_frame(
    df: pd.DataFrame,
    target_hz: float,
    time_col: str = "timestamp",
    max_gap_ms: float = 200.0,
) -> pd.DataFrame:
    """
    把对齐后的数据重采样到固定频率, **按连续段分别处理**。

    关键: 先在大于 max_gap_ms 的间隙处把数据切开, 每段独立重采样, 再拼回。
    因此停顿间隙会被完整保留, 不会被线性插值"补"成虚构数据
    (早期版本对整段插值, 一个 2755 秒的会话被撑成 137,747 行, 几乎全是编的)。

    数值列线性插值; 布尔/类别列 (label / fixStatus / gps_valid 等) 取最近邻。
    """
    if target_hz is None or target_hz <= 0 or len(df) < 2:
        return df

    t = df[time_col].to_numpy(dtype=np.float64)
    breaks = np.where(np.diff(t) > max_gap_ms)[0] + 1
    bounds = np.concatenate([[0], breaks, [len(df)]])

    parts = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:            # 段内样本太少, 不足以插值
            if b > a:
                parts.append(df.iloc[a:b])
            continue
        parts.append(_resample_uniform(df.iloc[a:b], target_hz, time_col))

    if not parts:
        return df
    return pd.concat(parts, ignore_index=True)


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
        target_hz: float = 50.0,
        min_source_hz: float = 20.0,
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
        self.target_hz = target_hz
        self.min_source_hz = min_source_hz

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

            # Step A2: 重采样到固定频率 — 让 "窗口=150 帧" 有确定的物理时长
            src_hz = estimate_rate_hz(session.imu["timestamp"])
            # 200ms: 超过 4 个 50Hz 周期即视为真实断点, 不插值、不跨段切窗。
            # 实测 1000ms 会把一个 15875 行的会话撑到 57961 行 (3.7x, 大半是编的),
            # 200ms 则与原生采样率换算一致。
            seg_gap_ms = 200.0
            if self.target_hz and self.target_hz > 0:
                if src_hz > 0 and src_hz < self.min_source_hz:
                    if verbose:
                        print(f"  [{session.session_id}] 跳过: 原生采样率仅 "
                              f"{src_hz:.1f} Hz (< {self.min_source_hz} Hz), "
                              f"上采样会凭空插值")
                    continue
                n_before = len(aligned)
                aligned = resample_frame(
                    aligned, self.target_hz, max_gap_ms=seg_gap_ms
                )
                if verbose and src_hz > 0:
                    print(f"  [{session.session_id}] 重采样: {src_hz:.1f} Hz → "
                          f"{self.target_hz:.0f} Hz ({n_before} → {len(aligned)} 行)")

            # 标记连续段: 重采样只在段内进行, 切窗也只在段内进行
            aligned["_segment"] = assign_segments(
                aligned["timestamp"].values, seg_gap_ms
            )

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
            seg_arr = aligned["_segment"].values
            if len(seg_arr) != len(valid_mask):
                seg_arr = None
            else:
                seg_arr = seg_arr[valid_mask]

            # Step D: 逐 Session 做滑动窗口 (防止跨 Session 数据泄漏)
            session_label = LABEL_MAP.get(session.label.upper().strip(), -1)
            if session_label < 0:
                if verbose:
                    print(f"  [{session.session_id}] 跳过: 未知标签 '{session.label}'")
                continue

            session_X, session_Y = _sliding_windows_by_segment(
                feat, labels, seg_arr, window_size, stride,
                fixed_label=session_label,
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
    target_hz: float = 50.0,
    min_source_hz: float = 20.0,
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
        target_hz=target_hz,
        min_source_hz=min_source_hz,
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
    target_hz: float = 50.0,
    min_source_hz: float = 20.0,
) -> list[dict]:
    """
    逐 Session 构建窗口 (未标准化), 返回会话级窗口列表。

    用于 Session 级训练/验证划分: 标准化参数只能在训练会话上拟合。

    Args:
        target_hz:     重采样目标频率 (Hz); <=0 表示不重采样
        min_source_hz: 原生采样率低于此值的会话直接跳过 (避免上采样凭空插值)

    Returns:
        [{"session_id": str, "label": str, "label_id": int,
          "X": [n, window, D], "Y": [n], "n_frames": int}, ...]
    """
    ds = SessionDataset(data_dir, verbose=False,
                        repair_timestamps=repair_timestamps)
    out = []

    # 先探出特征维度, 再构造恒等 scaler (不改变数值, 只绕过未拟合检查)。
    # 探测时不重采样、不跳过, 保证任何会话都能给出维度。
    probe_session = ds.sessions[0]
    probe = MotionWindowDataset(
        sessions=[probe_session], window_size=window_size, stride=stride,
        rolling_window=rolling_window, max_gap_ms=max_gap_ms, builder=builder,
        drop_features=drop_features, target_hz=0,
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
        try:
            d = MotionWindowDataset(
                sessions=[s], window_size=window_size, stride=stride,
                scaler=ident, fit_scaler=False,
                rolling_window=rolling_window, max_gap_ms=max_gap_ms, builder=builder,
                drop_features=drop_features,
                target_hz=target_hz, min_source_hz=min_source_hz,
                verbose=False,
            )
        except ValueError:
            # 例如原生采样率过低被跳过 → 没有可用窗口
            print(f"  [跳过] {s.session_id}: 无可用的窗口 "
                  f"(采样率过低或帧数不足)")
            continue
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


def _pick_val_by_count(group, target, val_ratio, seed):
    """
    每类留出 round(n * val_ratio) 个会话 (至少 1, 且至少留 1 个给训练),
    在同样数量的候选组合中, 选窗口总数最接近 target 的那组。

    按会话数而非窗口数决定留出规模, 这样新增的小会话不会被全部吸进验证集;
    再按窗口数挑选具体哪些会话, 避免验证集被单个超大会话撑爆。
    """
    import itertools

    n = len(group)
    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1)

    # 稳定排序, 保证组合枚举与并列取舍是确定性的
    ordered = sorted(group, key=lambda p: p["session_id"])

    best, best_key = None, None
    for combo in itertools.combinations(range(n), n_val):
        win = sum(ordered[i]["X"].shape[0] for i in combo)
        key = (abs(win - target), tuple(ordered[i]["session_id"] for i in combo))
        if best_key is None or key < best_key:
            best_key, best = key, [ordered[i] for i in combo]
    return best


def _pick_val_by_size(group, target, seed):
    """旧策略: 按窗口数从小到大累加, 取最接近目标的前缀。"""
    rng = np.random.RandomState(seed)
    jitter = rng.rand(len(group))
    order = np.lexsort((jitter, [p["X"].shape[0] for p in group]))
    ordered = [group[i] for i in order]

    best_n, best_gap = 1, None
    acc = 0
    for n, p in enumerate(ordered[:-1], start=1):   # 至少留 1 个会话给训练
        acc += p["X"].shape[0]
        gap = abs(acc - target)
        if best_gap is None or gap < best_gap:
            best_gap, best_n = gap, n
    return ordered[:best_n]


def session_level_split(
    per_session: list[dict],
    val_ratio: float = 0.2,
    seed: int = 42,
    strategy: str = "count",
) -> tuple[list[dict], list[dict], list[str]]:
    """
    按 Session 分组划分训练/验证集, 每类至少留出 1 个会话。

    strategy:
      "count" (默认) — 每类按 **会话数** 留出 round(n * val_ratio) 个 (至少 1)。
                       在同样数量的候选中, 选窗口总数最接近目标的那组,
                       避免验证集全被小会话占据、也避免被超大会话撑爆。
      "size"         — 旧行为: 按窗口数从小到大累加, 取最接近目标的前缀。
                       缺点: 若新增的会话都比较小, 它们会**全部**被吸进验证集,
                       训练集一个新会话都拿不到 (实测 6 个新会话全部落入 val)。

    该类只有 1 个会话时无法留出, 全部进训练集。

    Returns:
        (train_parts, val_parts, val_session_ids)
    """
    if strategy not in ("count", "size"):
        raise ValueError(f"strategy 必须是 'count' 或 'size', 收到 {strategy!r}")

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

        total_win = sum(p["X"].shape[0] for p in group)
        target = total_win * val_ratio

        if strategy == "count":
            held = _pick_val_by_count(group, target, val_ratio, seed)
        else:
            held = _pick_val_by_size(group, target, seed)

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
