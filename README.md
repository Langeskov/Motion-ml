# Motion-ml

运动状态分类模型训练工程 — 基于 PyTorch，使用手机 IMU + GNSS 传感器数据。

## 支持的模型

| 模型 | 参数量 | 说明 |
|------|--------|------|
| `lstm` | ~210K | 双层 LSTM |
| `cnn_lstm` | ~297K | CNN 提取局部震动 + LSTM 时序建模 |
| `cnn_transformer` | ~409K | CNN + Transformer Encoder + Attention Pooling |
| `motion_encoder` | ~419K | 自监督预训练 Encoder，支持 pretrain → finetune |

## 运动类别

| ID | 类别 | 说明 |
|----|------|------|
| 0 | STATIONARY | 静止 |
| 1 | WALKING | 步行 |
| 2 | CYCLING | 骑行 |
| 3 | CAR | 汽车 |
| 4 | TRAIN | 火车/高铁 |

## 数据格式

MotionRecorder v2 双 CSV 格式，按文件名前缀自动配对：

```
data/raw/
├── CAR_20260916_143959_imu.csv     # 100~200 Hz
├── CAR_20260916_143959_gnss.csv    # 1~5 Hz
├── WALKING_20260912_173758_imu.csv
├── WALKING_20260912_173758_gnss.csv
└── ...
```

**IMU 列:** `timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, roll, pitch, yaw, label`

**GNSS 列:** `timestamp, latitude, longitude, altitude, speed, bearing, accuracy, satelliteCount, fixStatus`

## 数据管线

```
SessionDataset          扫描目录，配对 *_imu.csv ↔ *_gnss.csv
  → TimeAligner         GNSS speed 对齐到 IMU 时间轴 (≤2s)
  → FeatureBuilder      m/s→km/h, NaN→0, 输出 [N, 10]
  → FeatureExtractor    10维 → 21维 (模长、滚动统计、震动指标)
  → StandardScaler      标准化
  → Sliding Windows     窗口=150帧, 步长=50, 逐Session切分
  → MotionWindowDataset PyTorch Dataset
```

## 评估协议 (重要)

**不要用窗口级随机划分评估。** 滑动窗口 `stride=50 < window=150`，相邻窗口共享 100 帧；
窗口级 `random_split` 会让验证窗口与训练窗口在同一段采集上重叠。

实测（23 个 Session / 55,172 窗口）：

| 划分方式 | 验证窗口与训练窗口时间重叠 | 结论 |
|----------|---------------------------|------|
| 窗口级 random_split | **99.84%** | 验证指标 ≈ 训练集精度，完全不可信 |
| Session 级划分（默认） | **0%** | 验证集是模型从未见过的整段采集 |

同理，`evaluate.py` 不带 `--all` 时只评估 checkpoint 中记录的留出会话。

```bash
# 训练（默认 session 级划分，无泄漏）
python train.py --model cnn_transformer --epochs 60 \
    --split_mode session --select_metric macro_f1

# 评估（自动只评估留出会话）
python evaluate.py --checkpoint checkpoints/best_cnn_transformer.pth

# 评估全量数据（含训练集，指标虚高，仅用于冒烟）
python evaluate.py --checkpoint ... --all
```

其它注意事项：

- **选模型用 `--select_metric macro_f1`**。CAR 占 54.9% 的窗口，用 accuracy 选模型会掩盖某一类完全失效
  （实测 accuracy 83% 时 STATIONARY 召回为 0）。
- **绝对姿态角（yaw/pitch/roll）跨会话不泛化**。它们只反映手机朝向，与运动状态无关；
  实测两个 STATIONARY 会话的 `yaw` 标准化均值相差 2.84，模型因此把静止误判为 TRAIN。
  可用 `--drop_features yaw,pitch,roll` 剔除。

## 基线结果

23 个 Session（17 训练 / 6 留出）、Session 级划分、60 epoch、macro-F1 选模型：

| 配置 | 特征数 | Accuracy | Macro F1 | STATIONARY F1 | WALKING F1 | CYCLING F1 | CAR F1 | TRAIN F1 |
|------|-------:|---------:|---------:|--------------:|-----------:|-----------:|-------:|---------:|
| CNN-Transformer | 21 | 78.5% | 0.579 | 0.000 | 0.819 | 0.358 | 0.829 | 0.889 |
| 自监督预训练 + 微调 | 21 | 76.4% | 0.572 | 0.000 | 0.912 | 0.236 | 0.800 | 0.909 |
| CNN-Transformer（去 yaw） | 20 | 78.1% | 0.679 | 0.565 | 0.838 | 0.253 | 0.805 | 0.937 |
| CNN-Transformer（去 yaw/pitch/roll） | 18 | 77.8% | **0.690** | 0.769 | 0.697 | 0.285 | 0.805 | 0.891 |

结论：

- 两种模型在同等协议下**基本打平**（macro F1 0.579 vs 0.572），预训练没有带来可测量的泛化增益。
- 剔除绝对姿态角后 macro F1 提升 **+0.111**，STATIONARY 从完全失效恢复到 F1 0.769。
- 所有配置的 train accuracy 都在 98% 以上，而验证 76~79%，**过拟合明显**。
- CYCLING 的 precision 只有 0.16~0.36（大量 CAR 被判成 CYCLING），是下一个要解决的点。

## 已知数据问题

- **IMU 时间戳重复**：23 个会话全部存在，合计 22.1% 的行与其他行共享同一 `timestamp`，
  但传感器值互相不同（同时间戳最多 330 条、330 组不同加速度）——是采集端缓冲区批量写入导致的
  时间戳损坏，不是真重复。旧代码 `drop_duplicates` 会直接丢弃这些真实采样；
  现在默认按组内行序展开修复（`SessionDataset(repair_timestamps=True)`），保留全部样本。
  根治需要在采集端修时间戳。
- **类别覆盖不均**：CAR 54.9%、TRAIN 19.1%、WALKING 9.8%、CYCLING 8.7%、STATIONARY 7.5%。
  STATIONARY 只有 2 个会话，跨会话泛化无从验证；CYCLING 只有 4 个。
- **单会话样本量差距悬殊**：最大的 CAR 会话 15,810 窗口，最小的仅 45 窗口。

## 快速开始

```bash
pip install -r requirements.txt
```

### 有监督训练

```bash
# 推荐配置: session 级划分 + macro-F1 选模型 + 剔除绝对姿态角
python train.py --model cnn_transformer --epochs 60 \
    --split_mode session --select_metric macro_f1 \
    --drop_features yaw,pitch,roll

python train.py --model cnn_lstm --epochs 60 --split_mode session --select_metric macro_f1
python train.py --model lstm --epochs 60 --split_mode session --select_metric macro_f1
```

### 自监督预训练 → 微调

```bash
# Stage 1: 预训练 (只用训练会话, 不看留出会话)
python pretrain.py --epochs 30 --split_mode session

# Stage 2: 微调
python finetune.py --pretrained checkpoints/encoder_pretrained.pth \
    --epochs 60 --split_mode session --select_metric macro_f1
```

### 评估与导出

```bash
python evaluate.py --checkpoint checkpoints/best_cnn_transformer.pth
python export.py --checkpoint checkpoints/best_cnn_transformer.pth --onnx
```

### TensorBoard

```bash
tensorboard --logdir runs/
```

## 项目结构

```
motion-ml/
├── train.py                 # 有监督训练 (4种模型)
├── pretrain.py              # Stage 1: Masked Sensor Modeling 预训练
├── finetune.py              # Stage 2: 有监督微调
├── evaluate.py              # 模型评估
├── export.py                # 导出 .pth / ONNX
├── test_v5_pipeline.py      # 管线完整性测试
├── dataset/
│   ├── session_loader.py    # Session 发现与加载
│   ├── session_dataset.py   # 滑动窗口 Dataset
│   ├── time_aligner.py      # GNSS → IMU 时间对齐
│   ├── feature_builder.py   # 特征构建 + FeatureMap
│   └── preprocess.py        # FeatureExtractor (10→21维)
├── models/
│   ├── common.py            # PositionalEncoding, AttentionPooling
│   ├── motion_lstm.py       # LSTM
│   ├── cnn_lstm.py          # CNN-LSTM
│   ├── transformer.py       # CNN-Transformer
│   ├── motion_encoder.py    # MotionEncoder (预训练+微调)
│   └── mask_generator.py    # Mask 生成器
└── requirements.txt
```

## 传感器特征 (21维)

**基础 (10维):** gps_speed(km/h), acc_x/y/z, gyro_x/y/z, roll, pitch, yaw

**派生 (11维):** acc/gyro_magnitude, speed_delta, acc/gyro/roll 滚动均值+标准差, pitch_rolling_mean, vibration_score

## License

Private research project.