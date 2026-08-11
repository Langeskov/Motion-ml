# Motion-ml V5 — 运动状态识别模型训练工程

基于 PyTorch 的运动状态分类模型，支持 LSTM / CNN-LSTM / CNN-Transformer / 自监督预训练。

使用手机 IMU + GNSS 传感器数据，通过 MotionRecorder 双 CSV 采集。

## 项目结构

```
motion-ml/
├── data/raw/                   # MotionRecorder 双 CSV (imu + gnss)
├── dataset/
│   ├── session_loader.py       # Session 发现、IMU/GNSS 配对加载
│   ├── time_aligner.py         # GNSS → IMU 时间对齐
│   ├── feature_builder.py      # 单位转换(m/s→km/h)、NaN处理、FeatureMap
│   ├── preprocess.py           # FeatureExtractor (10→21维派生特征)
│   ├── session_dataset.py      # MotionWindowDataset + build_session_dataset
│   ├── loader.py               # (旧) 单 CSV 兼容接口
│   └── __init__.py
├── models/
│   ├── motion_lstm.py          # 纯 LSTM
│   ├── cnn_lstm.py             # CNN-LSTM 混合
│   ├── transformer.py          # CNN-Transformer 混合
│   ├── motion_encoder.py       # 自监督 Encoder + Pretrain/Classifier
│   ├── mask_generator.py       # Masked Sensor Modeling
│   └── __init__.py             # 模型工厂
├── train.py                    # 训练脚本
├── pretrain.py                 # Stage 1: 自监督预训练
├── finetune.py                 # Stage 2: 有监督微调
├── evaluate.py                 # 评估脚本
├── export.py                   # 模型导出
├── test_v5_pipeline.py         # 完整管线验证
├── requirements.txt
└── README.md
```

## 数据格式

MotionRecorder v2 输出 **双 CSV** 格式，按前缀自动配对：

```
data/raw/
├── CYCLING_20260807_083227_imu.csv   (125 Hz, 10 列)
└── CYCLING_20260807_083227_gnss.csv  (1 Hz, 9 列)
```

**IMU 列:** `timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, roll, pitch, yaw, label`

**GNSS 列:** `timestamp, latitude, longitude, altitude, speed, bearing, accuracy, satelliteCount, fixStatus`

## 数据管线

```
SessionDataset
  → scan directory, match *_imu.csv ↔ *_gnss.csv by prefix
  ↓
TimeAligner
  → GNSS speed (m/s) 对齐到每条 IMU (最近时间戳, ≤2秒)
  ↓
FeatureBuilder
  → 单位转换: gps_speed m/s ×3.6 → km/h
  → NaN → 0, 输出 [N, 10] + FeatureMap (name↔index)
  ↓
FeatureExtractor
  → 10维 → 21维: 模长、滚动统计、震动指标等
  ↓
StandardScaler → 滑动窗口 (150帧, stride=50) → MotionWindowDataset
```

### 实际数据测试结果

| 指标 | CYCLING_20260807_083227 |
|------|--------------------------|
| IMU 原始 | 110,568 行 |
| IMU 清洗后 | 50,972 行 (去重+排序) |
| GNSS 原始 | 1,416 行 |
| IMU 采样率 | ~125 Hz |
| GNSS 采样率 | ~1.0 Hz |
| GNSS 对齐率 | 100% (50,972/50,972) |
| GNSS 最大偏移 | 928 ms, 中位数 248 ms |
| gps_speed 均值 | 3.69 m/s → 13.28 km/h |
| 滑动窗口数 | 1,017 |
| 窗口形状 | (1017, 150, 21) |

## 传感器特征 (21 维)

### 基础特征 (10 维, FeatureBuilder 输出)

| 索引 | 特征 | 单位 | 说明 |
|------|------|------|------|
| 0 | gps_speed | km/h | GNSS 速度 (FeatureBuilder 从 m/s 转换) |
| 1 | acc_x | m/s² | 加速度 X 轴 |
| 2 | acc_y | m/s² | 加速度 Y 轴 |
| 3 | acc_z | m/s² | 加速度 Z 轴 |
| 4 | gyro_x | rad/s | 陀螺仪 X 轴 |
| 5 | gyro_y | rad/s | 陀螺仪 Y 轴 |
| 6 | gyro_z | rad/s | 陀螺仪 Z 轴 |
| 7 | roll | ° | 横滚角 |
| 8 | pitch | ° | 俯仰角 |
| 9 | yaw | ° | 偏航角 |

### 派生特征 (11 维, FeatureExtractor 生成)

| 特征 | 说明 |
|------|------|
| acc_magnitude | 加速度模长 √(x²+y²+z²) |
| gyro_magnitude | 陀螺仪模长 |
| speed_delta | 相邻帧速度差 |
| acc_rolling_mean/std | 加速度滚动统计 (10帧窗口) |
| gyro_rolling_mean/std | 陀螺仪滚动统计 (10帧窗口) |
| roll_rolling_mean/std | Roll 滚动统计 |
| pitch_rolling_mean | Pitch 滚动均值 |
| vibration_score | 震动强度 (变异系数) |

## 运动类别

| 类别 | ID | 说明 |
|------|----|------|
| STATIONARY | 0 | 静止 |
| WALKING | 1 | 行走 |
| CYCLING | 2 | 电动自行车 |
| CAR | 3 | 汽车 |
| TRAIN | 4 | 火车/高速 |

## 快速开始

### 1. 环境

```bash
pip install -r requirements.txt   # torch, numpy, pandas, scikit-learn, matplotlib, tensorboard
```

### 2. 准备数据

```bash
data/raw/
├── CYCLING_YYYYMMDD_HHMMSS_imu.csv
├── CYCLING_YYYYMMDD_HHMMSS_gnss.csv
├── WALKING_YYYYMMDD_HHMMSS_imu.csv
├── WALKING_YYYYMMDD_HHMMSS_gnss.csv
...
```

### 3. 训练 (有监督)

```bash
# LSTM (默认)
python train.py --model lstm --epochs 50

# CNN-LSTM
python train.py --model cnn_lstm --epochs 50

# CNN-Transformer
python train.py --model cnn_transformer --epochs 50
```

### 4. 自监督预训练 + 微调

```bash
# Stage 1: 预训练 (无需标签)
python pretrain.py --data_dir data/raw --epochs 50

# Stage 2: 微调
python finetune.py --data_dir data/raw --pretrained checkpoints/encoder_pretrained.pth --epochs 100
```

### 5. 评估

```bash
python evaluate.py --checkpoint checkpoints/best_lstm.pth
```

输出: Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report

### 6. 导出

```bash
python export.py --checkpoint checkpoints/best_lstm.pth
python export.py --checkpoint checkpoints/best_lstm.pth --onnx  # ONNX
```

## FeatureMap (按名访问, 不依赖列号)

```python
from dataset.feature_builder import FeatureBuilder, FeatureMap, FUTURE_SPECS

# 默认 10 维
builder = FeatureBuilder()
fm = builder.feature_map
gps_idx = fm.index_of("gps_speed")  # 0

# 扩展未来的传感器 (barometer, mag, hdop, optical_flow...)
builder = FeatureBuilder(extra_specs=FUTURE_SPECS)
```

## 模型对比

| 模型 | 参数数量 | 特点 |
|------|---------|------|
| LSTM | ~210K | 时序建模，简单高效 |
| CNN-LSTM | ~297K | CNN 提取局部震动模式 + LSTM 时序 |
| CNN-Transformer | ~409K | 局部 CNN + 全局 Attention |
| MotionEncoder (自监督) | ~419K | 预训练 + 微调，适合少标签场景 |

## TensorBoard

```bash
tensorboard --logdir runs/
```

## 输出目录

```
checkpoints/
├── best_lstm.pth                # LSTM 最优
├── best_cnn_lstm.pth            # CNN-LSTM 最优
├── encoder_pretrained.pth       # 自监督预训练 Encoder
└── motion_classifier.pth        # 微调后的分类器
```

## 已知局限 & 待验证项

> **说明:** 本章节仅定义实验方案，不修改当前已验证的 V5 数据管线。
> 当前基线保持: 10D Base Features → 21D FeatureExtractor → StandardScaler → (150, 21) → Model。

### 1. GNSS 丢星与 NaN→0 的已知局限

当前 `FeatureBuilder` 对 GNSS 无效数据执行 `NaN → 0 km/h`:

```
TimeAligner: GNSS 超时 (>2s) 或无有效 fix
  ↓
gps_speed = NaN
  ↓
FeatureBuilder: NaN → 0
  ↓
gps_speed = 0 km/h
```

**潜在风险:** 模型无法仅通过 `gps_speed` 判断 "设备真实静止" 还是 "设备正在运动，但 GNSS 暂时丢失"。

| 场景 | 真实状态 | 模型看到的 `gps_speed` | 潜在风险 |
|------|---------|------------------------|---------|
| 隧道内车辆行驶 | 正在运动 | 0 km/h | 模型可能将运动误判为静止 |
| 地下车库低速行驶 | 正在运动 | 0 km/h | IMU 仍有震动但与速度 0 矛盾 |
| 室内骑行 (无 GPS) | 正在运动 | 0 km/h | 完全依赖 IMU, 无速度参照 |
| 高楼遮挡路段 | 正在运动 | 0 km/h | 短时间内速度特征跳变 |
| GNSS 信号暂时丢失 | 正在运动 | 0 km/h | 间歇性速度缺失 |
| GNSS 长时间无有效 Fix | 正在运动或静止 | 0 km/h | 无法区分运动/静止 |

以上均为待验证的潜在风险，并非已确认的分类错误。

### 2. 方案 A：当前基线 (NaN → 0 km/h)

**当前生产/实验基线:**

```
gps_speed = NaN  →  gps_speed = 0 km/h
```

**优点:**
- 实现简单，已验证通过
- 不改变当前 10D FeatureMap
- 不改变现有 21D FeatureExtractor
- 不需要修改现有模型输入结构 (150, 21)

**缺点:**
- GNSS 丢失状态与真实 0 km/h 无法区分
- 在 GNSS 长时间丢失环境下可能产生误导特征
- 模型可能学到 "gps_speed=0 → STATIONARY" 的错误关联

### 3. 实验方案 B：NaN → Mask (gps_valid)

将 GNSS 有效性显式作为模型输入，让模型自行学习是否信任速度值。

**新增特征:**

| 索引 | 特征 | 单位 | 说明 |
|------|------|------|------|
| 10 | gps_valid | binary | 1 = GNSS 当前有效, 0 = GNSS 无效/缺失 |

`gps_valid` 不替代 `gps_speed`，两者同时存在:

```
# GNSS 有效时:
gps_speed = 5.39 km/h
gps_valid = 1

# GNSS 无效时:
gps_speed = 0 km/h   ← 占位值
gps_valid = 0        ← 标记为无效
```

这样模型可以区分:

| 情况 | gps_speed | gps_valid | 含义 |
|------|-----------|-----------|------|
| 真实静止 | 0 | 1 | 设备静止，GNSS 正常 |
| GNSS 丢失 (运动中) | 0 | 0 | 速度未知，GNSS 失效 |
| 正常行驶 | 5.39 | 1 | 速度可信，GNSS 正常 |

**输入维度变化:**

```
原: 10D Base Features → 21D FeatureExtractor → Model
新: 11D Base Features → 对应派生特征 → Model (维度 = 待验证)
```

**FeatureMap 仍然负责特征名称和索引管理:**

```python
from dataset.feature_builder import FeatureSpec, FeatureBuilder, DEFAULT_SPECS

# 方案 B: 追加 gps_valid
extended_specs = list(DEFAULT_SPECS) + [
    FeatureSpec(name="gps_valid", source_col="gps_valid"),
]
builder_b = FeatureBuilder(specs=extended_specs)

# 按名访问，不依赖列号
fm = builder_b.feature_map
assert fm.index_of("gps_speed") == 0
assert fm.index_of("gps_valid") == 10
```

### 4. 实验设计 (四步骤)

以下实验**仅在实验分支进行**，不修改当前 V5 基线。

#### Step 1 — 构造 GNSS 丢星测试集

从已有 Session 中动态构造人工 GNSS blackout，不修改 `data/raw/` 原始文件。

**方法:** 在 preprocessing / test pipeline 中，对指定时间窗口屏蔽 GNSS 数据（删除 `gnss` DataFrame 中对应行或标记 `gps_valid=False`）。

**测试不同 blackout duration:**

| Duration | 场景模拟 |
|----------|---------|
| 2 s | 短暂遮挡 |
| 5 s | 中等遮挡 |
| 10 s | 长隧道/地下通道 |
| 30 s | 长时间 GNSS 失效 |

#### Step 2 — 训练两个模型

保持完全相同条件，仅比较 baseline vs mask:

| 条件 | 保持一致 |
|------|---------|
| Session 划分 | 相同 train/val split |
| Random Seed | 固定 |
| Window Size | 150 |
| Stride | 50 |
| Optimizer | Adam, 相同 lr |
| Epoch / Batch Size | 相同 |
| StandardScaler | 各自独立 fit |
| Model Architecture | 相同 (input_dim 由 FeatureMap 自动适配) |

**推荐:** 以 **CNN-LSTM** 为主实验模型。如资源允许，可用 **LSTM** 做交叉验证。

#### Step 3 — 对比指标

除整体 Accuracy 外，重点关注以下指标:

| 指标 | 说明 |
|------|------|
| Accuracy | 整体正确率 |
| Macro F1 | 类别均衡 F1 |
| Weighted Precision / Recall | 加权精确率/召回率 |
| Confusion Matrix | 类间混淆 (重点: STATIONARY ↔ CAR, STATIONARY ↔ CYCLING) |
| Per-duration Accuracy | 按 blackout duration (2/5/10/30s) 分组评估 |

**重点观察:**
- GNSS blackout 时间增加后，Baseline 与 Mask 方案的性能差异变化趋势
- `gps_valid=False` 窗口子集上的准确率对比

**不要只报告总体 Accuracy。** 必须分别展示各 blackout duration 下的性能。

#### Step 4 — 代码示例 (概念性)

```python
# ── Baseline: NaN → 0 (当前) ──
gps_speed_nan  = float('nan')
gps_speed_fill = 0.0            # FeatureBuilder 填充为 0
# 模型看到: gps_speed=0.0 (无法区分 真实静止 vs GNSS丢失)

# ── Mask: NaN → 0 + gps_valid ──
# GNSS 有效时:
gps_speed = 5.39
gps_valid = 1

# GNSS 无效时:
gps_speed = 0        # 占位
gps_valid = 0        # 标记无效

# ── FeatureMap 验证 ──
from dataset.feature_builder import FeatureBuilder, FeatureSpec, DEFAULT_SPECS

extended = list(DEFAULT_SPECS) + [
    FeatureSpec(name="gps_valid", source_col="gps_valid"),
]
builder = FeatureBuilder(specs=extended)
fm = builder.feature_map
fm.index_of("gps_speed")  # → 0
fm.index_of("gps_valid")  # → 10
```

### 5. 状态标记

| 项目 | 状态 |
|------|------|
| 当前 V5 基线 | 使用 `NaN → 0` (10D) |
| `gps_valid` 加入默认模型 | **未实施** |
| FeatureMap 扩展机制 | **已预留** |
| 本章节实验方案 | **仅定义，待实施** |
| 完成实验后评估是否升级管线 | **待定** |

**注意:** 本章节不修改当前 README 中已有的 10D / 21D / (150, 21) 数据流程描述。
实验仅在独立分支进行，当前主线管线保持不变。
