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
  → Resample            重采样到固定 50 Hz (按连续段分别处理, 间隙保留)
  → FeatureBuilder      m/s→km/h, NaN→0, 输出 [N, 10]
  → FeatureExtractor    10维 → 21维 (模长、滚动统计、震动指标)
  → StandardScaler      标准化
  → Sliding Windows     窗口=150帧(=3秒), 步长=50, 段内切分
  → MotionWindowDataset PyTorch Dataset
```

**为什么必须重采样**：各会话原生 IMU 采样率实测覆盖 4.9 ~ 250 Hz，
若按帧数切窗，同样 150 帧在不同会话上代表 0.6 ~ 30.8 秒，物理时长相差 50 倍，
滚动统计特征的含义也随之改变。统一到 50 Hz 后「150 帧 = 3 秒」才有确定含义。

**为什么按段重采样**：录音中存在最长 525 秒的停顿。若对整段做线性插值，
会把停顿"补"成虚构数据（实测一个 15,875 行的会话被撑到 57,961 行，3.7 倍）。
现在以 200 ms 为界切段，段内重采样、段内切窗，停顿完整保留，窗口也不会跨越中断。
原生采样率低于 20 Hz 的会话（如 4.9 Hz 的那个）直接跳过，避免上采样凭空插值。


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

### 正则化消融与运行间方差

单次运行的消融（dropout / weight_decay / 数据增强）看起来有差异，但重复 3 个随机种子后
可以看出这些差异**全部落在运行间波动之内**：

| 配置 | Accuracy (mean±std, n=3) | Macro F1 (mean±std, n=3) | 三次运行最优 epoch |
|------|--------------------------|--------------------------|--------------------|
| 基线 dp0.3 | 0.798 ± 0.037 | 0.704 ± 0.011 | 4 / 2 / 35 |
| dp0.5 | 0.817 ± 0.034 | 0.737 ± 0.030 | 12 / 49 / 5 |

要点：

- **同一配置换种子，准确率波动 ±3.5pt**。任何小于这个量级的"改进"都不可信，
  单次实验的配置对比没有意义 —— 必须多种子重复。
- 最优 epoch 集中在 **前 4~12 轮**（60 轮训练里 80~90% 的算力是浪费的）。
  早停能显著加速实验，且不会降低结果。
- 结论：**继续调正则化/训练技巧的收益已经饱和**，瓶颈在数据覆盖（见下节）。

### 重采样 A/B（锁定同一组留出会话）

同一划分、同一特征集（18 维，去 yaw/pitch/roll）、同一 `--seed 42`，只切换 `--target_hz`：

| 配置 | 训练窗口 | 验证窗口 | Accuracy | Macro F1 | STATIONARY F1 | WALKING F1 | CYCLING F1 | CAR F1 | TRAIN F1 | 最优 epoch |
|------|---------:|---------:|---------:|---------:|--------------:|-----------:|-----------:|-------:|---------:|-----------:|
| `--target_hz 0`（不重采样） | 60,785 | 11,145 | 83.7% | 0.684 | **0.000** | 0.861 | 0.807 | 0.805 | 0.947 | 31/60 |
| `--target_hz 50`（默认） | 18,187 | 4,991 | 79.6% | 0.679 | **0.578** | 0.492 | 0.558 | 0.880 | 0.887 | **2/60** |

要点：

- Macro F1 基本持平（0.684 vs 0.679），**但错误分布完全不同**：
  不重采样时 STATIONARY 在留出会话上 193 个窗口**全部判错**；重采样后同一会话
  188 个窗口判对 147 个（78%）。
- 重采样后训练快 **3 倍**（4.9 s/epoch vs 15.4 s/epoch），因为窗口数按物理时长对齐后
  从 60,785 降到 18,187。
- 两个配置的最优 epoch 分别在第 31 和第 2 轮 —— 再次说明早停是必要的。
- 结论：重采样修掉了一个真实缺陷（时间尺度不一致），代价是 WALKING 的 F1 下滑，
  净效应在噪声内。保留重采样是出于**正确性**而非精度，同时顺带获得 3 倍加速。

### Session 级 K 折交叉验证（当前最可信的评估）

单次留出划分的问题是：macro F1 里每一类其实只由 1~2 个会话决定，换个划分就大幅跳动。
改成按会话分 K 折后，每类会话轮流做验证集，**所有会话都被评到恰好一次**，
每类指标建立在全部会话之上。

```bash
python cv.py --folds 3 --target_hz 50 --drop_features yaw,pitch,roll
```

3 折袋外（out-of-fold）结果，23,178 个样本：

| 类别 | F1 | support（窗口数） |
|------|-----|------------------:|
| STATIONARY | 0.587 | 1,073 |
| WALKING | 0.476 | 2,233 |
| CYCLING | 0.519 | 1,824 |
| CAR | 0.912 | 14,017 |
| TRAIN | 0.814 | 4,031 |
| **总体** | **Accuracy 0.812 / Macro F1 0.662** | 23,178 |

主要混淆（行=真实）：

| 真实 → 误判 | 数量 | 占该类 |
|-------------|-----:|-------:|
| TRAIN → STATIONARY | 760 | 18.9% |
| CYCLING → CAR | 834 | 45.7% |
| CAR → CYCLING | 530 | 3.8% |
| WALKING → CAR | 576 | 25.8% |
| WALKING → TRAIN | 478 | 21.4% |

要点：

- **OOF macro F1 0.662 低于单次划分的 0.679~0.690** —— 说明单次划分的结果偏乐观，
  它恰好抽到了比较好评的会话组合。以 0.662 作为当前真实水平的基准。
- CYCLING 与 CAR 的双向混淆是最大问题（45.7% 的骑行被判成汽车）。
- 开早停后 3 折总共只花 **136 秒**（每折最优 epoch 分别为 14/14/1），
  所以现在做多种子、多模型的对比变得很便宜。

## 已知数据问题

- **IMU 时间戳重复**：23 个会话全部存在，合计 22.1% 的行与其他行共享同一 `timestamp`，
  但传感器值互相不同（同时间戳最多 330 条、330 组不同加速度）——是采集端缓冲区批量写入导致的
  时间戳损坏，不是真重复。旧代码 `drop_duplicates` 会直接丢弃这些真实采样；
  现在默认按组内行序展开修复（`SessionDataset(repair_timestamps=True)`），保留全部样本。
  根治需要在采集端修时间戳。
- **类别覆盖不均**：CAR 占 54.9% 的窗口，但会话数分布极不均衡。按当前划分（9 训练 / 14 留出）：

  | 类别 | 总会话 | 训练 | 留出 |
  |------|-------:|-----:|-----:|
  | STATIONARY | 2 | **1** | 1 |
  | WALKING | 3 | 2 | 1 |
  | CYCLING | 4 | 2 | 2 |
  | CAR | 9 | **2** | 7 |
  | TRAIN | 5 | 2 | 3 |
  | 合计 | 23 | 9 | 14 |

  STATIONARY 在训练集里只有 **1 个会话**，留出集也只有 1 个（315 窗口）——
  这就是 macro F1 里 STATIONARY 项在 0.44~0.77 之间乱跳的原因：它完全由单个会话决定。
  CYCLING 的 precision 长期偏低（0.16~0.37）也源于训练会话太少。

- **单会话样本量差距悬殊**：最大的 CAR 会话 15,810 窗口，最小的仅 45 窗口。

- **采样率不一致（当前最大隐患）**：各会话的 IMU 有效采样率相差悬殊，
  导致同样 150 帧的窗口对应的**真实时长**从 0.6 秒到 30 秒不等：

  | 会话 | 有效采样率 | 150 帧 = |
  |------|-----------:|---------:|
  | STATIONARY_20261022_103430 | 4.9 Hz | **30.8 s** |
  | STATIONARY_20261022_100043 | 47.6 Hz | 3.1 s |
  | 2026-10 新采集的 WALKING/CYCLING | 58.8 Hz | 2.6 s |
  | 2026-09 之前的多数会话 | 166~250 Hz | 0.6~0.9 s |

  这意味着 `stride=50` 的滑动窗口在不同会话上代表完全不同的物理时间尺度，
  滚动统计特征（10 帧窗口）的含义也随之改变。这是跨会话泛化差的重要原因之一，
  且与模型结构无关。

  已实现：`--target_hz`（默认 50）在特征工程前统一重采样，`--min_source_hz`（默认 20）
  跳过原生采样率过低的会话；分段与切窗细节见上文「数据管线」。

## 采集优先级建议

当前瓶颈是数据覆盖，不是模型或训练技巧（见上节：正则化收益已落在噪声内）。
建议按以下顺序补充采集，目标每类 ≥5 个训练会话：

| 优先级 | 类别 | 建议补充 | 覆盖场景 |
|--------|------|---------|---------|
| 1 | STATIONARY | 3~5 个会话 | 办公室静坐、车内静止、地铁/火车停站、手机放桌面 vs 口袋 |
| 2 | CYCLING | 2~3 个会话 | 不同路况（平坦/颠簸）、不同车型/助力模式 |
| 3 | WALKING | 2~3 个会话 | 不同步速、上下楼梯、手持 vs 口袋 |
| 4 | TRAIN | 1~2 个会话 | 不同车型（高铁/普速），含进出站加减速 |
| 5 | CAR | 暂缓 | 已有 9 个会话，相对充足 |

采集时注意：每次会话保持单一运动状态、单一手机放置方式，便于按会话切分；
并确保采集端写入正确的单调递增 timestamp（见上节时间戳问题）。


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
├── cv.py                    # Session 级 K 折交叉验证 (推荐的可信评估)
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