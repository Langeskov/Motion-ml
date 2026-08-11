"""
FeatureBuilder
==============

将已对齐的 Session 数据转换为神经网络输入特征。

职责 (唯一):
  1. 单位转换: GNSS speed m/s → km/h
  2. 缺失值处理: NaN → 0
  3. 特征名 → 列索引映射 (FeatureMap)

设计原则:
  - 不修改原始 CSV
  - 不修改 aligned DataFrame (返回新的 numpy 数组)
  - 模型代码通过 FeatureMap 按名访问特征，不依赖固定列号
  - 新增传感器只需注册 FeatureSpec，无需修改下游代码

数据流:
  TimeAligner 输出 (含 gps_speed m/s)
    → FeatureBuilder.build()
    → (features [N, D], FeatureMap)

用法:
    builder = FeatureBuilder()
    features, feature_map = builder.build(aligned)

    # 下游代码按名取索引:
    gps_idx = feature_map.index_of("gps_speed")
    acc_x_idx = feature_map.index_of("acc_x")

    # 或使用 feature_map.columns 构建 DataFrame:
    df = pd.DataFrame(features, columns=feature_map.columns)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── 单个传感器规格 ──────────────────────────────────────────

@dataclass
class FeatureSpec:
    """
    单个传感器特征规格。

    Attributes:
        name:         特征名 (如 "gps_speed")
        source_col:   在 aligned DataFrame 中的来源列名 (如 "gps_speed")
        unit_from:    原始单位 (如 "m/s")
        unit_to:      目标单位 (如 "km/h")
        scale_factor: 转换系数 (unit_to = unit_from * factor)
        default_val:  缺失时填充值
    """
    name: str
    source_col: str = ""
    unit_from: str = ""
    unit_to: str = ""
    scale_factor: float = 1.0
    default_val: float = 0.0

    def __post_init__(self):
        if not self.source_col:
            self.source_col = self.name


# ── FeatureMap: 特征名 ↔ 列索引 ────────────────────────────

@dataclass
class FeatureMap:
    """
    特征名 → 列索引映射。

    模型训练代码通过此对象按名访问特征，不依赖固定列号。

    属性:
        specs:   FeatureSpec 列表 (决定特征顺序)
        _index:  {name: column_index} 快速查找表
    """
    specs: list[FeatureSpec] = field(default_factory=list)
    _index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self._rebuild()

    def _rebuild(self):
        """根据 specs 重建 _index 映射表。"""
        self._index = {s.name: i for i, s in enumerate(self.specs)}

    def index_of(self, name: str) -> int:
        """
        返回特征名对应的列索引。

        Raises:
            KeyError: 特征名不存在
        """
        return self._index[name]

    def get_index(self, name: str, default: int = -1) -> int:
        """安全取索引，不存在返回 default。"""
        return self._index.get(name, default)

    @property
    def columns(self) -> list[str]:
        """特征列名列表 (按顺序)。"""
        return [s.name for s in self.specs]

    @property
    def feature_dim(self) -> int:
        """特征维度。"""
        return len(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __repr__(self) -> str:
        lines = [f"FeatureMap(dim={self.feature_dim})"]
        for spec in self.specs:
            transform = ""
            if spec.scale_factor != 1.0:
                transform = f" [{spec.unit_from}→{spec.unit_to} ×{spec.scale_factor}]"
            lines.append(f"  [{self.index_of(spec.name):>2d}] {spec.name}{transform}")
        return "\n".join(lines)


# ── 当前默认特征规格 (V1 — 10维) ──────────────────────────

DEFAULT_SPECS: list[FeatureSpec] = [
    # GNSS — 速度 (m/s → km/h)
    FeatureSpec(
        name="gps_speed",
        source_col="gps_speed",
        unit_from="m/s",
        unit_to="km/h",
        scale_factor=3.6,
        default_val=0.0,
    ),
    # IMU — 加速度计 (m/s²，无转换)
    FeatureSpec(name="acc_x"),
    FeatureSpec(name="acc_y"),
    FeatureSpec(name="acc_z"),
    # IMU — 陀螺仪 (rad/s，无转换)
    FeatureSpec(name="gyro_x"),
    FeatureSpec(name="gyro_y"),
    FeatureSpec(name="gyro_z"),
    # IMU — 欧拉角 (度，无转换)
    FeatureSpec(name="roll"),
    FeatureSpec(name="pitch"),
    FeatureSpec(name="yaw"),
]

# ── 未来扩展规格 (V2+，按需启用) ────────────────────────────

FUTURE_SPECS: list[FeatureSpec] = [
    # GPS 附加字段
    FeatureSpec(name="gps_accuracy",     source_col="gps_accuracy"),
    FeatureSpec(name="gps_bearing",      source_col="bearing"),
    FeatureSpec(name="gps_satellite",    source_col="satelliteCount"),
    FeatureSpec(name="hdop",             source_col="hdop"),
    # 气压计
    FeatureSpec(name="barometer",        source_col="pressure"),
    # 磁力计
    FeatureSpec(name="mag_x",            source_col="mag_x"),
    FeatureSpec(name="mag_y",            source_col="mag_y"),
    FeatureSpec(name="mag_z",            source_col="mag_z"),
    # 光流速度
    FeatureSpec(name="optical_flow_speed", source_col="optical_flow_speed"),
]

# 向后兼容: 当前 10 维特征名列表
OUTPUT_COLS = [s.name for s in DEFAULT_SPECS]


# ── BuildStats ──────────────────────────────────────────────

@dataclass
class BuildStats:
    """构建统计信息。"""
    total_rows: int = 0
    gps_valid_rows: int = 0
    gps_invalid_rows: int = 0
    gps_mean_ms: float = 0.0
    gps_mean_kmh: float = 0.0
    nan_filled: int = 0

    def __repr__(self) -> str:
        return (
            f"BuildStats(rows={self.total_rows}, gps_valid={self.gps_valid_rows}, "
            f"gps_invalid={self.gps_invalid_rows}, "
            f"mean_ms={self.gps_mean_ms:.2f}, mean_kmh={self.gps_mean_kmh:.2f})"
        )


# ── FeatureBuilder ──────────────────────────────────────────

class FeatureBuilder:
    """
    神经网络输入特征构建器。

    Args:
        specs:          特征规格列表 (默认 DEFAULT_SPECS, 10维)
        extra_specs:    额外追加的特征规格 (用于扩展)
    """

    def __init__(
        self,
        specs: Optional[list[FeatureSpec]] = None,
        extra_specs: Optional[list[FeatureSpec]] = None,
    ):
        chosen = list(specs) if specs else list(DEFAULT_SPECS)
        if extra_specs:
            chosen.extend(extra_specs)
        self.feature_map = FeatureMap(specs=chosen)
        self._gps_spec = next(
            (s for s in chosen if s.name == "gps_speed"), None
        )

    @property
    def feature_dim(self) -> int:
        return self.feature_map.feature_dim

    def build(
        self,
        aligned: pd.DataFrame,
    ) -> tuple[np.ndarray, FeatureMap, BuildStats]:
        """
        从 TimeAligner 输出的对齐数据构建特征矩阵。

        Args:
            aligned: TimeAligner.align() 返回的 DataFrame

        Returns:
            (features, feature_map, stats)
            features:     [N, D] float64 numpy 数组
            feature_map:  特征名→列索引 映射 (与构建时配置一致)
            stats:        BuildStats 统计
        """
        stats = BuildStats(total_rows=len(aligned))
        specs = self.feature_map.specs
        feature_arrays = []

        for spec in specs:
            col = spec.source_col
            arr: np.ndarray

            if col in aligned.columns:
                arr = aligned[col].values.astype(np.float64)
            else:
                # 列不存在 → 全 0
                arr = np.zeros(stats.total_rows, dtype=np.float64)

            # 单位转换
            if spec.scale_factor != 1.0:
                arr = arr * spec.scale_factor

            # 记录转换前后的 GPS 速度统计
            if spec.name == "gps_speed":
                valid_mask = aligned.get("gps_valid", pd.Series([True] * len(aligned)))
                valid_src = aligned.loc[valid_mask, spec.source_col].dropna() if spec.source_col in aligned.columns else pd.Series()
                if len(valid_src) > 0:
                    stats.gps_mean_ms = float(valid_src.mean())
                    stats.gps_mean_kmh = stats.gps_mean_ms * spec.scale_factor

            # 缺失值处理
            nan_count = int(np.isnan(arr).sum())
            stats.nan_filled += nan_count
            arr = np.nan_to_num(arr, nan=spec.default_val)

            feature_arrays.append(arr)

        # 统计 gps_valid
        if "gps_valid" in aligned.columns:
            stats.gps_valid_rows = int(aligned["gps_valid"].sum())
            stats.gps_invalid_rows = stats.total_rows - stats.gps_valid_rows

        features = np.column_stack(feature_arrays)

        return features, self.feature_map, stats
