# Motion-ml

基于手机 IMU + GNSS 传感器的运动状态分类（静止 / 步行 / 骑行 / 汽车 / 火车）。
PyTorch 训练，ONNX 部署到移动端。

**当前指标**（3 折 × 6 种子会话级交叉验证，132,252 个袋外样本）：

| Accuracy | Macro F1 | 总错误率 |
|---------:|---------:|---------:|
| **86.7% ± 1.8%** | **0.785 ± 0.033** | 13.3% |

> 为什么用 CV 而不是单次训练的 val 数字：同配置单次留出的波动是 0.683~0.753，
> 比模型之间的差异还大。详见「评估协议」。

## 快速开始

```bash
pip install -r requirements.txt

# 训练 (推荐配置)
python train.py --model motion_encoder --epochs 60 --batch_size 64 \
    --split_mode session --select_metric macro_f1 --target_hz 50 \
    --drop_features yaw,pitch,roll

# 交叉验证 —— 衡量真实泛化能力 (推荐)
python cv.py --folds 3 --target_hz 50 --drop_features yaw,pitch,roll

# 评估单个 checkpoint (只跑留出会话)
python evaluate.py --checkpoint checkpoints/best_motion_encoder.pth

# 导出 ONNX 给移动端 + 验证
python export.py --checkpoint checkpoints/best_motion_encoder.pth --onnx
python verify_onnx.py

# 测试
python test_pipeline.py
```

## 运动类别

| ID | 类别 | 说明 |
|----|------|------|
| 0 | STATIONARY | 静止 |
| 1 | WALKING | 步行 |
| 2 | CYCLING | 电动车骑行 |
| 3 | CAR | 汽车 |
| 4 | TRAIN | 火车/高铁 |

## 数据格式

MotionRecorder v2 双 CSV，按文件名前缀自动配对：

```
data/raw/
├── CAR_20260916_143959_imu.csv     # 100~200 Hz
├── CAR_20260916_143959_gnss.csv    # 1~5 Hz
├── WALKING_20261024_143731_imu.csv
├── WALKING_20261024_143731_gnss.csv
└── ...
```

**IMU 列：** `timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, roll, pitch, yaw, label`
（2026-10 之后的采集还多一列 `imu_ts_ns` 审计列，不影响读取）

**GNSS 列：** `timestamp, latitude, longitude, altitude, speed, bearing, accuracy, satelliteCount, fixStatus`

## 数据管线

```
SessionDataset          扫描目录，配对 *_imu.csv ↔ *_gnss.csv
  → TimeAligner         GNSS speed 对齐到 IMU 时间轴 (≤2s)
  → Resample            重采样到固定 50 Hz (按连续段分别处理, 间隙保留)
  → FeatureBuilder      m/s→km/h, NaN→0, 输出 [N, 10]
  → FeatureExtractor    10维 → 27维 (模长、滚动统计、震动指标、频带能量/谱质心/主频)
  → StandardScaler      标准化 (只在训练会话上拟合)
  → Sliding Windows     窗口=150帧(=3秒), 步长=50, 段内切分
  → MotionWindowDataset PyTorch Dataset
```

**为什么必须重采样**：各会话原生采样率实测覆盖 4.9 ~ 250 Hz。若按帧数切窗，
同样 150 帧在不同会话上代表 0.6 ~ 30.8 秒，物理时长相差 50 倍。
统一到 50 Hz 后「150 帧 = 3 秒」才有确定含义。

**为什么按段重采样**：录音中存在最长 525 秒的停顿。若对整段做线性插值会把停顿
"补"成虚构数据（实测一个 15,875 行的会话被撑到 57,961 行，3.7 倍）。现在以 200 ms
为界切段，段内重采样、段内切窗，停顿完整保留，窗口也不会跨越中断。
原生采样率低于 20 Hz 的会话直接跳过，避免上采样凭空插值。

## 传感器特征 (27 维)

| 组 | 维度 | 内容 |
|----|-----:|------|
| 基础 | 10 | gps_speed(km/h), acc_x/y/z, gyro_x/y/z, roll, pitch, yaw |
| 时域派生 | 11 | acc/gyro_magnitude, speed_delta, acc/gyro/roll 滚动均值+标准差, pitch_rolling_mean, vibration_score |
| 频域派生 | 6 | acc_bp_0.5_2 / 2_5 / 5_10 / 10_25（各频带能量占比）, acc_centroid, acc_dom_freq |

默认训练用 `--drop_features yaw,pitch,roll` 剔除绝对姿态角，**实际输入 24 维**。
姿态角只反映手机朝向、跨会话不泛化（实测两个 STATIONARY 会话的 yaw 差 2.84σ，
会让模型把静止判成火车）。

## 评估协议

**不要用窗口级随机划分。** 滑动窗口 `stride=50 < window=150`，相邻窗口共享 100 帧，
窗口级 `random_split` 会让验证窗口与训练窗口在同一段采集上重叠。实测（23 会话 / 55,172 窗口）：

| 划分方式 | 验证窗口与训练窗口时间重叠 | 结论 |
|----------|---------------------------|------|
| 窗口级 random_split | **99.84%** | 验证指标 ≈ 训练集精度 |
| Session 级划分（默认） | **0%** | 验证集是模型没见过的整段采集 |

因此：

- `train.py` 默认 `--split_mode session`，留出整段会话
- `evaluate.py` 只评估 checkpoint 里记录的留出会话，**不提供全量评估**（全量包含训练样本）
- 衡量真实泛化请用 `cv.py`：按会话分 K 折，每类会话轮流做验证，汇总袋外指标

选模型用 `--select_metric macro_f1` 而不是 accuracy。CAR 占 54.9% 的窗口，
用 accuracy 会掩盖某一类完全失效（实测 accuracy 83% 时 STATIONARY 召回为 0）。

**CV 的折分只依赖会话与种子**，所以不同特征/超参配置用同一 `--seed` 就得到相同的折分，
可以做干净的配对对比。这是本项目所有结论的实验方法基础。

## 结果

### 每类表现（6 种子 × 3 折，132,252 袋外样本）

| 类别 | F1 | 标准差 | 样本数 |
|------|-----:|-------:|-------:|
| CAR | 0.915 | 0.014 | 84,102 |
| TRAIN | 0.866 | 0.047 | 24,186 |
| STATIONARY | 0.752 | 0.102 | 6,438 |
| CYCLING | 0.710 | 0.081 | 10,944 |
| WALKING | 0.683 | 0.117 | 6,582 |

累计混淆矩阵（行=真实，列=预测）：

| | STATIONARY | WALKING | CYCLING | CAR | TRAIN |
|---|---:|---:|---:|---:|---:|
| **STATIONARY** | 5,257 | 0 | 23 | 154 | 1,004 |
| **WALKING** | 0 | 4,210 | 236 | 2,112 | 24 |
| **CYCLING** | 19 | 110 | 8,941 | 1,761 | 113 |
| **CAR** | 599 | 1,390 | 4,822 | 75,613 | 1,678 |
| **TRAIN** | 1,795 | 33 | 157 | 1,559 | 20,642 |

主要错误块：`CAR→CYCLING` 4,822（占汽车 5.7%）、`WALKING→CAR` 2,112（占步行 32.1%）、
`TRAIN→STATIONARY` 1,795（7.4%）。

### 模型对比

| 模型 | 参数量 | OOF macro F1 |
|------|-------:|-------------:|
| **`motion_encoder`** | 104,006 | **0.785** ← 推荐 |
| `cnn_transformer` | 126,918 | 0.713 |
| `cnn_lstm` | 87,237 | 0.695 |
| `lstm` | 56,645 | 0.686 |

> 参数量按 `hidden_size=64`（当前默认）、24 维输入计。后三个数字是 `hidden_size=128`
> 时测的，仅用于相对排序（未在 64 下复测）。

### 指标可信度

单次运行的波动是 **±0.015~0.025 macro F1**（同配置换种子）。
任何小于这个量级的"改进"都不成立 —— 结论必须基于多种子配对对比，且带标准差。
完整的实验演进与排除项见 [`docs/experiments.md`](docs/experiments.md)。

## 移动端部署 (ONNX)

```bash
python export.py --checkpoint checkpoints/best_motion_encoder.pth --output_dir exported --onnx
python verify_onnx.py        # 必须跑, 导出成功 ≠ 正确
```

产物 `exported/motion_motion_encoder.onnx`（约 0.6 MB，单文件，权重内嵌）：

| 属性 | 值 |
|------|-----|
| 输入 | `input`, float32, `[batch_size, 150, 24]`（batch 动态） |
| 输出 | `output`, float32, `[batch_size, 5]`（batch 动态） |
| opset | 18 |

导出验证（`verify_onnx.py`）：PyTorch vs ONNX max\|Δ\| = 2.38e-06（batch 1/2/3/8/32）；
22,042 个真实窗口预测一致率 **100.000000%**。

**端上必须做标准化**，输入是 24 维派生特征而非原始传感器值：

```python
x = (raw_features - scaler_mean) / scaler_scale   # raw_features: [150, 24]
probs = session.run(None, {"input": x[None]})[0]  # [1, 5]
```

`scaler_mean` / `scaler_scale` 存在 checkpoint 里（`exported/*.pth` 也带），
用错或漏掉这步，输出就是垃圾。窗口 150 帧 = **3 秒**（采样率固定 50 Hz）。

> **导出坑**：torch 2.x 的 `torch.onnx.export` 默认走 dynamo 导出器，遇到
> `nn.TransformerEncoder` 会把 batch 维钉死成 1（`dynamic_shapes` 怎么写都没用）。
> `export.py` 强制 `dynamo=False` 走 TorchScript 导出器才能保住动态 batch。
> 改了模型结构务必用 `verify_onnx.py` 复验。

## 项目结构

```
motion-ml/
├── train.py                 # 训练 (4 种模型, session 级划分, 早停, EMA, 数据增强)
├── cv.py                    # Session 级 K 折交叉验证 (推荐的可信评估)
├── evaluate.py              # 评估单个 checkpoint (只跑留出会话)
├── export.py                # 导出 .pth / ONNX
├── verify_onnx.py           # ONNX 导出验证 (数值一致性 / 动态 batch)
├── test_pipeline.py         # 管线完整性测试
├── dataset/
│   ├── session_loader.py    # Session 发现、IMU/GNSS 配对加载、时间戳修复
│   ├── session_dataset.py   # 重采样、滑动窗口、会话级划分
│   ├── time_aligner.py      # GNSS → IMU 时间对齐
│   ├── feature_builder.py   # 单位转换 + FeatureMap
│   └── preprocess.py        # FeatureExtractor (10 → 27 维)
├── models/
│   ├── common.py            # PositionalEncoding, AttentionPooling
│   ├── motion_encoder.py    # motion_encoder (推荐)
│   ├── transformer.py       # cnn_transformer
│   ├── cnn_lstm.py          # cnn_lstm
│   └── motion_lstm.py       # lstm
├── docs/experiments.md      # 实验演进记录
└── requirements.txt
```

## 已知数据问题

- **IMU 时间戳重复（2026-10 之前的采集）**：22.1% 的行与其他行共享 `timestamp`
  但传感器值互不相同 —— 采集端缓冲区批量写入导致的时间戳损坏，不是真重复。
  现在按组内行序展开修复（`repair_timestamps=True`），保留全部样本。
  2026-10 之后重新采集的会话已彻底修复（`dup=0`、严格单调递增）。
  **根治需要在采集端修**：每个采样点用事件自带的高精度时钟打时间戳，不要在落盘时统一赋值。
- **录音中存在长停顿**：最长 525 秒。管线已按段切分处理，但采集时最好一停就分段存新文件。
- **类别覆盖不均**：各类别可用会话数 STATIONARY 3、WALKING 4、CYCLING 6、CAR 9、TRAIN 5。
  STATIONARY 与 WALKING 的 F1 标准差最大（±0.10~0.12）正是因此。
- **采样率不一致**：原生 4.9~250 Hz，已由重采样统一到 50 Hz。但 4.9 Hz 的会话
  （`STATIONARY_20261022_103430`）无法上采样，被跳过。

## 采集优先级建议

当前瓶颈是数据覆盖而非模型（训练侧优化已系统性排查并饱和）。目标每类 ≥5 个可用会话：

| 优先级 | 类别 | 建议补充 | 覆盖场景 |
|--------|------|---------|---------|
| 1 | STATIONARY | 3~5 个会话 | 办公室静坐、车内静止、地铁停站、手机放桌面 vs 口袋 |
| 2 | WALKING | 2~3 个会话 | 不同步速、上下楼、手持 vs 口袋 |
| 3 | CYCLING | 1~2 个会话 | 不同路况、不同车型 |
| 4 | TRAIN | 1~2 个会话 | 不同车型，含进出站加减速 |
| 5 | CAR | 暂缓 | 已有 9 个会话 |

采集要求：每次会话保持单一运动状态、单一手机放置方式；确保写入单调递增的时间戳。

## License

Private research project.
