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

## 快速开始

```bash
pip install -r requirements.txt
```

### 有监督训练

```bash
python train.py --model cnn_transformer --epochs 50
python train.py --model cnn_lstm --epochs 50
python train.py --model lstm --epochs 50
```

### 自监督预训练 → 微调

```bash
# Stage 1: 预训练 (无需标签)
python pretrain.py --epochs 50

# Stage 2: 微调
python finetune.py --pretrained checkpoints/encoder_pretrained.pth --epochs 100
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