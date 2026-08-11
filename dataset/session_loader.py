"""
MotionRecorder v2 Session Dataset Loader
=========================================

自动发现并加载双CSV格式的采集数据。

文件命名规则:
  {label}_{timestamp}_imu.csv   (100~200Hz)
  {label}_{timestamp}_gnss.csv  (1~5Hz)

同一前缀自动配对为一个 Session，无需手工指定。

数据流:
  scan directory → match prefix → load pair → SessionDataset
"""

import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


# ── GNSS fixStatus 有效值 ──────────────────────────────────
VALID_FIX_STATUSES = {"LOCKED", "ACTIVE"}


@dataclass
class Session:
    """
    单次采集会话。

    Attributes:
        session_id: 唯一标识，格式 "{label}_{timestamp}"
        label:      运动类别 (STATIONARY / WALKING / CYCLING / CAR / TRAIN)
        imu:        IMU 传感器 DataFrame (已清洗)
        gnss:       GNSS 定位 DataFrame (已清洗)
        imu_path:   原始 IMU 文件路径
        gnss_path:  原始 GNSS 文件路径
    """
    session_id: str
    label: str
    imu: pd.DataFrame
    gnss: pd.DataFrame
    imu_path: str = ""
    gnss_path: str = ""

    def __repr__(self) -> str:
        return (
            f"Session(id={self.session_id!r}, label={self.label!r}, "
            f"imu_rows={len(self.imu)}, gnss_rows={len(self.gnss)})"
        )


@dataclass
class SessionInfo:
    """扫描阶段的元数据，不含实际数据。"""
    prefix: str
    label: str
    timestamp: str
    imu_path: str
    gnss_path: Optional[str] = None


# ── 文件名解析 ─────────────────────────────────────────────

# 匹配: LABEL_DATETIME_imu.csv 或 LABEL_DATETIME_gnss.csv
_FILENAME_RE = re.compile(
    r"^(?P<label>[A-Z]+)_(?P<timestamp>\d{8}_\d{6})_(?P<sensor>imu|gnss)\.csv$"
)


def _parse_filename(filename: str) -> Optional[dict]:
    """解析文件名，返回 {label, timestamp, sensor} 或 None。"""
    m = _FILENAME_RE.match(filename)
    if m is None:
        return None
    return m.groupdict()


def _build_prefix(label: str, timestamp: str) -> str:
    return f"{label}_{timestamp}"


# ── 扫描与配对 ─────────────────────────────────────────────

def scan_sessions(data_dir: str) -> list[SessionInfo]:
    """
    扫描目录，寻找 *_imu.csv 并尝试匹配同名 *_gnss.csv。

    Args:
        data_dir: 数据目录路径

    Returns:
        SessionInfo 列表，每个元素描述一个可加载的 Session
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"目录中没有 CSV 文件: {data_dir}")

    # 按 prefix 分组
    imu_map: dict[str, SessionInfo] = {}
    gnss_map: dict[str, str] = {}  # prefix → gnss_path

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        parsed = _parse_filename(fname)
        if parsed is None:
            continue

        prefix = _build_prefix(parsed["label"], parsed["timestamp"])

        if parsed["sensor"] == "imu":
            imu_map[prefix] = SessionInfo(
                prefix=prefix,
                label=parsed["label"],
                timestamp=parsed["timestamp"],
                imu_path=fpath,
            )
        elif parsed["sensor"] == "gnss":
            gnss_map[prefix] = fpath

    # 配对: 以 imu 为主，gnss 为辅
    sessions: list[SessionInfo] = []
    for prefix, info in imu_map.items():
        info.gnss_path = gnss_map.get(prefix)
        sessions.append(info)

    if not sessions:
        raise FileNotFoundError(
            f"未找到 *_imu.csv 文件: {data_dir}\n"
            f"文件命名应为: LABEL_YYYYMMDD_HHMMSS_imu.csv"
        )

    return sessions


# ── 数据清洗 ────────────────────────────────────────────────

def _clean_imu(df: pd.DataFrame) -> pd.DataFrame:
    """
    IMU 数据清洗:
    1. 删除重复 timestamp (保留最后一条)
    2. 按 timestamp 排序
    3. 前向填充 + 后向填充缺失值
    4. 删除仍残留 NaN 的行
    """
    before = len(df)

    # 去重: 同一 timestamp 可能因传感器回调重复
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    # 排序
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 填充传感器列
    sensor_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                   "roll", "pitch", "yaw"]
    existing_sensor_cols = [c for c in sensor_cols if c in df.columns]
    df[existing_sensor_cols] = df[existing_sensor_cols].ffill().bfill()

    # 删除残留 NaN
    df = df.dropna(subset=existing_sensor_cols).reset_index(drop=True)

    after = len(df)
    if before != after:
        print(f"  [IMU]  cleaned: {before} → {after} rows")
    return df


def _clean_gnss(df: pd.DataFrame) -> pd.DataFrame:
    """
    GNSS 数据清洗:
    1. 删除无效 fix (fixStatus 不在有效集)
    2. 删除 lat=0 & lon=0 的记录
    3. 按 timestamp 排序
    4. 前向填充缺失值 (GNSS 采样率低，允许插值)
    """
    before = len(df)

    # 删除无效 fixStatus
    if "fixStatus" in df.columns:
        df = df[df["fixStatus"].isin(VALID_FIX_STATUSES)]

    # 删除无效坐标
    if "latitude" in df.columns and "longitude" in df.columns:
        invalid = (df["latitude"] == 0) & (df["longitude"] == 0)
        df = df[~invalid]

    # 排序
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 前向填充 (GNSS 数据允许)
    gnss_cols = ["latitude", "longitude", "altitude", "speed", "bearing", "accuracy"]
    existing_gnss_cols = [c for c in gnss_cols if c in df.columns]
    if existing_gnss_cols:
        df[existing_gnss_cols] = df[existing_gnss_cols].ffill().bfill()

    after = len(df)
    if before != after:
        print(f"  [GNSS] cleaned: {before} → {after} rows")
    return df


# ── 加载 ────────────────────────────────────────────────────

def _load_imu(path: str) -> pd.DataFrame:
    """加载并清洗 IMU CSV。"""
    df = pd.read_csv(path)
    df = _clean_imu(df)
    return df


def _load_gnss(path: Optional[str]) -> pd.DataFrame:
    """加载并清洗 GNSS CSV。如果路径为 None，返回空 DataFrame。"""
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = _clean_gnss(df)
    return df


def load_session(info: SessionInfo) -> Session:
    """
    根据 SessionInfo 加载一个完整 Session。

    Args:
        info: 扫描阶段生成的元数据

    Returns:
        包含 imu/gnss DataFrame 的 Session 对象
    """
    imu_df = _load_imu(info.imu_path)
    gnss_df = _load_gnss(info.gnss_path)

    return Session(
        session_id=info.prefix,
        label=info.label,
        imu=imu_df,
        gnss=gnss_df,
        imu_path=info.imu_path,
        gnss_path=info.gnss_path or "",
    )


# ── SessionDataset ──────────────────────────────────────────

class SessionDataset:
    """
    多 Session 数据集。

    自动扫描目录，加载所有可配对的 Session。

    用法:
        ds = SessionDataset("data/raw")
        for session in ds.sessions:
            print(session.session_id, len(session.imu), len(session.gnss))

    Args:
        data_dir:  数据目录
        verbose:   是否打印加载日志
    """

    def __init__(self, data_dir: str, verbose: bool = True):
        self.data_dir = data_dir
        self.verbose = verbose

        # 扫描
        infos = scan_sessions(data_dir)
        if verbose:
            print(f"[SessionDataset] 发现 {len(infos)} 个 Session 候选")

        # 加载
        self.sessions: list[Session] = []
        for info in infos:
            session = load_session(info)
            self.sessions.append(session)
            if verbose:
                gnss_status = f"gnss={len(session.gnss)}" if len(session.gnss) > 0 else "gnss=无"
                print(f"  ✓ {session.session_id}: imu={len(session.imu)}, {gnss_status}")

        if verbose:
            labels = [s.label for s in self.sessions]
            unique_labels = sorted(set(labels))
            print(f"[SessionDataset] 加载完成: {len(self.sessions)} 个 Session, "
                  f"类别={unique_labels}")

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, idx: int) -> Session:
        return self.sessions[idx]

    def __repr__(self) -> str:
        return f"SessionDataset(dir={self.data_dir!r}, sessions={len(self.sessions)})"

    @property
    def labels(self) -> list[str]:
        """所有 Session 的标签列表。"""
        return [s.label for s in self.sessions]

    @property
    def session_ids(self) -> list[str]:
        """所有 Session 的 ID 列表。"""
        return [s.session_id for s in self.sessions]

    def filter_by_label(self, label: str) -> list[Session]:
        """按标签筛选 Session。"""
        return [s for s in self.sessions if s.label == label]

    def summary(self) -> str:
        """返回数据集摘要。"""
        lines = [f"SessionDataset: {len(self.sessions)} sessions"]
        from collections import Counter
        counts = Counter(self.labels)
        for label, count in sorted(counts.items()):
            lines.append(f"  {label}: {count}")
        return "\n".join(lines)
