# Motion-ml

运动状态分类模型训练工程 — 基于 PyTorch，使用手机 IMU + GNSS 传感器数据。

## 支持的模型

| 模型 | 参数量 | 说明 | OOF macro F1 |
|------|-------:|------|-------------:|
| **`motion_encoder`** | 104,006 | Transformer Encoder + AttentionPooling | **0.785** ← 推荐 |
| `cnn_transformer` | 126,918 | CNN + Transformer Encoder + AttentionPooling | 0.713 |
| `cnn_lstm` | 87,237 | CNN 提取局部震动 + LSTM 时序建模 | 0.695 |
| `lstm` | 56,645 | 双层 LSTM | 0.686 |

> 参数量按 `hidden_size=64`（当前默认）、24 维输入计。
> OOF macro F1 为 3 折 CV 均值；其中 `motion_encoder` 的 0.785 是 6 种子、
> 剔除脏会话 + hidden_size 64 的最终口径，其余三个是 hidden_size 128 的旧口径，
> 仅用于模型间相对排序（未在 h=64 下复测）。完整对比见「模型对比」一节。

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
  → FeatureExtractor    10维 → 27维 (模长、滚动统计、震动指标、频带能量/谱质心/主频)
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
python evaluate.py --checkpoint checkpoints/best_motion_encoder.pth

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

> 以下各节按时间顺序记录演进过程。**当前基准：`motion_encoder` + 24 维特征（含频域）
> + 剔除脏会话 + hidden_size 64，OOF accuracy 86.7% ± 1.8% / macro F1 0.785 ± 0.033**
> （见「当前基准」一节）。早期表格保留用于对照，但它们早于重采样、早停、频域特征和减容，
> 不可直接与最新数字比较。

### 阶段一：早期结果（23 会话，未重采样，单次留出）

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

3 折袋外（out-of-fold）结果，23,178 个样本
（**18 维特征、尚无频域特征**，作为后续改进的起点）：

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

### 频域特征的收益（配对实验）

给 `FeatureExtractor` 新增 6 维频域特征（4 个频带能量占比 + 谱质心 + 主频），
特征从 18 维升到 24 维。由于 `cv.py` 的折分只依赖会话与种子，**两种配置的折分完全相同**，
所以这是干净的配对对比（3 个种子 × 3 折，每个种子下基线与新特征用同一折分）：

| seed | 基线 18 维 | +频域 24 维 | 增量 |
|-----:|-----------:|------------:|-----:|
| 1 | 0.5759 | 0.6791 | +0.103 |
| 2 | 0.5457 | 0.7377 | +0.192 |
| 3 | 0.6314 | 0.7207 | +0.089 |
| **均值** | **0.5843 ± 0.0355** | **0.7125 ± 0.0246** | **+0.128** |

Accuracy：0.7614 → 0.8238（+0.062）。三次增量全部为正，且都远超运行间波动。

每类 F1（三种子均值）：

| 类别 | 基线 | +频域 | 增量 |
|------|-----:|------:|-----:|
| STATIONARY | 0.392 | 0.726 | **+0.334** |
| WALKING | 0.386 | 0.473 | +0.088 |
| CYCLING | 0.554 | 0.666 | **+0.112** |
| CAR | 0.891 | 0.909 | +0.019 |
| TRAIN | 0.700 | 0.789 | +0.088 |

五类全部提升。STATIONARY 与 CYCLING 涨得最多，正好对应诊断中指出的
"电动车振动集中在 2~5 Hz、汽车偏 10~25 Hz" 这一信号。

### 模型对比（24 维特征，3 折 × 3 种子）

| 模型 | Macro F1 | Accuracy | 单次 CV 耗时 |
|------|---------:|---------:|-------------:|
| **motion_encoder** | **0.7469 ± 0.0109** | 0.8365 ± 0.0154 | 126 s |
| cnn_transformer | 0.7125 ± 0.0246 | 0.8238 ± 0.0262 | 132 s |
| cnn_lstm | 0.6947 ± 0.0220 | 0.8189 ± 0.0196 | 79 s |
| lstm | 0.6855 ± 0.0145 | 0.8160 ± 0.0104 | 91 s |

每类 F1（三种子均值）：

| 类别 | motion_encoder | cnn_transformer | cnn_lstm | lstm |
|------|---------------:|----------------:|---------:|-----:|
| STATIONARY | 0.734 | 0.726 | 0.734 | 0.648 |
| WALKING | **0.590** | 0.473 | 0.404 | 0.419 |
| CYCLING | **0.693** | 0.666 | 0.651 | 0.686 |
| CAR | 0.916 | 0.909 | 0.911 | 0.912 |
| TRAIN | **0.803** | 0.789 | 0.774 | 0.763 |

结论：

- **`motion_encoder`（MotionClassifier 架构）全面领先**，且三种子逐一对比都赢
  （+0.054 / +0.010 / +0.039，均值 +0.034）。
- 它的运行间波动也是最小的（±0.011 vs cnn_transformer 的 ±0.025）。
- 它在 WALKING 上优势最明显（0.590 vs 0.473）—— 随机游走式的姿态变化更依赖全局注意力。
- **推荐默认模型：`motion_encoder`**。

### 训练侧优化的系统性排查（结论：已饱和）

针对"错误率高"逐项试了常规优化手段，**全部用配对 CV（3 种子、同折分）验证，
没有任何一项超过现状**：

| 配置 | seed 1 | seed 2 | seed 3 | 均值 |
|------|-------:|-------:|-------:|-----:|
| A 恒定 lr（现状） | 0.7327 | 0.7144 | 0.7521 | **0.733** |
| B +cosine warmup | 0.7221 | 0.6986 | 0.7405 | 0.720 |
| C +cosine+EMA 权重 | 0.7437 | 0.7119 | 0.7104 | 0.722 |
| D +数据增强+dropout0.5+wd1e-4 | 0.7327 | 0.7140 | 0.7228 | 0.723 |

要点：

- warmup **确实**把"最优 epoch"从第 1~3 轮推到第 8~31 轮（说明早期峰值是真过拟合，
  不是偶然噪声），但推后之后 macro F1 并没有更高。
- 数据增强、更强 dropout、L2 正则、权重 EMA —— 全部无效。
- **单次运行的波动是 ±0.015~0.025**，任何小于这个量级的"改进"都不成立。

> 附带修掉一个实现 bug：EMA 原先按 epoch 更新一次，而 decay=0.999 的有效记忆约 1000 步，
> 导致 60 轮只推进约 6%，权重长期停留在初始化附近（macro F1 一度掉到 0.21）。
> 已改为每个 step 更新。

### 数据修正：一个标签错误的会话

排查各类混淆时发现 **`WALKING_20260912_173758` 并非步行**：

| 会话 | acc_std | gyro_mean | 平均速度 |
|------|--------:|----------:|---------:|
| WALKING_20260809_180043 | 2.73 | 0.85 | 4.7 km/h |
| WALKING_20261024_* | 3.55~3.87 | 1.75~2.13 | 4.3 km/h |
| **WALKING_20260912_173758** | **0.30** | **0.037** | **26.3 km/h** |

震动幅度只有正常步行的 1/9、陀螺仪 1/25，速度 26 km/h 步行不可能；
特征反而接近 `TRAIN_20260816_120745`（gyro 0.033）。它有 3,812 个窗口，是步行类最大会话。

移入 `data/quarantine/` 后（不猜它究竟是汽车还是地铁，直接隔离）：

| | 总错误率 | 备注 |
|---|---:|---|
| 排除前 | 18.1% | WALKING→TRAIN 38.1% |
| 排除后 | **16.0%** | 该混淆消失，露出真正的 WALKING→CAR 44.5% |

同时 CYCLING F1 +0.060、TRAIN F1 +0.052、accuracy +2.1pt。

> 也加了 `--exclude_sessions` 开关，以后发现异常会话可随时隔离而不必改动代码。

### 容量：hidden_size 128 会过拟合

排除脏会话后继续压容量（同样配对 CV，3 种子）：

| hidden_size | seed 1 | seed 2 | seed 3 | 均值 | 标准差 |
|------------:|-------:|-------:|-------:|-----:|-------:|
| 128 | 0.7354 | 0.7408 | 0.7504 | 0.742 | 0.006 |
| **64** | **0.7767** | 0.7588 | 0.7468 | **0.761** | 0.012 |
| 32 | 0.7579 | 0.7595 | 0.7569 | 0.758 | **0.001** |

- h=32 与 h=128 的逐种子增量 **+0.022 / +0.019 / +0.007 全为正**，趋势确凿。
- h=64 与 h=32 打平（0.761 vs 0.758），但 h=32 方差只有 ±0.001。
- 参数量从 408,710 降到 104,006。
- **默认 `--hidden_size` 已改为 64。**

### 当前基准

**`motion_encoder` + 24 维特征 + 剔除脏会话 + hidden_size 64：OOF macro F1 0.785 ± 0.033**

（演进轨迹：加频域前 0.584 → 加频域后 0.713 → 剔脏会话+减容 0.761 → 当前默认配置 0.785。
 最初那个"98.2% / 99.6%"是训练集精度，不在此列。）

最终口径：6 个种子 × 3 折 CV，共 132,252 个袋外样本。

| 指标 | 数值 | 标准差 |
|------|-----:|-------:|
| **Accuracy** | **0.867** | ±0.018 |
| **Macro F1** | **0.785** | ±0.033 |
| 总错误率 | 13.3% | |

每类 F1：

| 类别 | F1 | 标准差 | 累计样本 |
|------|-----:|-------:|--------:|
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

剩余的主要错误块：`CAR→CYCLING` 4,822（占汽车 5.7%）、`WALKING→CAR` 2,112（占步行 32.1%）、
`TRAIN→STATIONARY` 1,795（7.4%）。

**重要：不要用单次训练的 val 数字当"最终准确率"。** 同配置单次留出实测 0.683~0.753，
而 CV 是 0.785 —— 单次留出会抽到较难或较易的会话组合，波动远大于模型间差异。
任何结论都应基于 CV 且带 ± 标准差。

### 当前产物与复现

仓库内的 `checkpoints/best_motion_encoder.pth` 由以下命令生成（确定性模式，第 6 轮达峰、
第 16 轮早停，104,006 参数，val macro F1 0.700 / acc 75.3%）：

```bash
python train.py --model motion_encoder --epochs 60 --batch_size 64 \
    --split_mode session --select_metric macro_f1 --target_hz 50 \
    --drop_features yaw,pitch,roll --seed 42 --deterministic \
    --early_stop_patience 10 --val_sessions "STATIONARY_20261022_100043,\
WALKING_20260809_180043,CYCLING_20260807_083227,CAR_20260913_112250,\
CAR_20260916_143959,TRAIN_20260816_120745"
```

注意该 checkpoint 的 val 指标只基于 6 个留出会话，**波动很大**（同配置在别的划分上
可达 0.74~0.78）；**评估模型能力请用 `cv.py` 的袋外 macro F1（0.761）**，
而不是这个 val 数字。同理，单次运行的结果不要与 CV 数字直接比较。

## 已知数据问题

- **IMU 时间戳重复（已修复）**：2026-10 之前采集的 23 个会话全部存在该问题，合计 22.1% 的行
  与其他行共享同一 `timestamp`，但传感器值互相不同（同时间戳最多 330 条、330 组不同加速度）
  ——是采集端缓冲区批量写入导致的时间戳损坏，不是真重复。旧代码 `drop_duplicates` 会直接丢弃
  这些真实采样；现在默认按组内行序展开修复（`SessionDataset(repair_timestamps=True)`），保留全部样本。
  **2026-10 之后重新采集的 6 个会话已彻底修复：`dup=0`、严格单调递增**，不再需要修复逻辑。
- **类别覆盖不均**：CAR 占比最大的窗口。当前 28 个会话（另有 1 个被隔离，见下）：

  | 类别 | 总会话 | 可用（采样率 ≥20Hz、标签可信） |
  |------|-------:|-----------------------------:|
  | STATIONARY | 4 | 3 |
  | WALKING | 5 | **4**（1 个标签错误已隔离） |
  | CYCLING | 6 | 6 |
  | CAR | 9 | 9 |
  | TRAIN | 5 | 5 |
  | 合计 | 29 | 27 |

  仍偏少：STATIONARY 可用会话只有 3 个（其中一个 4.9 Hz 的已被跳过）。
  这正是 STATIONARY 的 F1 在不同划分间从 0.00 跳到 0.77 的原因 —— 它由极少数会话决定。
  改用 K 折交叉验证后（见上节），该类指标建立在全部 3 个会话、1,073 个窗口上，才具备参考价值。

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

## 诊断：CYCLING ↔ CAR 混淆

CV 结果里最大的一处错误是 **45.7% 的骑行被判成汽车**（834/1,824 窗口）。
针对这一点做了特征可分性诊断（6 个骑行 / 9 个汽车会话，重采样 50 Hz，窗口 150 帧）。

按会话聚合的均值（± 会话间标准差）：

| 特征 | CYCLING（电动车） | CAR | 差异 |
|------|------------------:|----:|------|
| `acc_std`（震动幅度） | **1.27 ± 0.37** | 0.45 ± 0.22 | 2.8× |
| `acc_ptp`（峰峰值） | **7.62 ± 2.19** | 2.62 ± 1.16 | 2.9× |
| `gyro_mean` | **0.45 ± 0.16** | 0.15 ± 0.10 | 3.0× |
| 频带 2–5 Hz 能量占比 | **0.56 ± 0.11** | 0.24 ± 0.07 | 踩踏/电机主频 |
| 频带 10–25 Hz 能量占比 | 0.08 ± 0.02 | **0.35 ± 0.15** | 汽车更接近宽带噪声 |
| 谱质心 | **5.2 ± 0.4 Hz** | 9.0 ± 2.6 Hz | |
| 主频 | **4.0 ± 0.7 Hz** | 7.5 ± 2.7 Hz | |
| 平均速度 | 4.1 ± 1.3 km/h | 5.8 ± 3.6 km/h | **区分力最弱** |

可分性验证（会话级 4 折分层分组，逻辑回归）：

| 特征组合 | Accuracy | Balanced Accuracy |
|----------|---------:|------------------:|
| 只用速度 | 0.700 | 0.753 |
| 只用时域震动 | 0.860 | 0.810 |
| 只用频域 | 0.732 | 0.801 |
| 速度 + 时域 + 频域 | 0.850 | **0.887** |

结论：

- 电动车的振动特征**在数据上非常清楚**：能量集中在 2–5 Hz（主频约 4 Hz），
  而汽车能量更分散、明显偏向 10–25 Hz，震动幅度只有骑行的三分之一。
- **速度是最弱的判别依据** —— 市区/拥堵路况的汽车均速只有 2–6 km/h，与骑行完全重叠。
- 一个简单线性模型用上频域特征就能到 0.887 balanced accuracy，
  而当前五分类模型有 45.7% 的骑行被误判 —— **信号本身足够可分，是模型没用上**。
- 当前 21 维特征**全部是时域统计，没有任何频域特征**。
  下一步优先给 `FeatureExtractor` 增加频带能量占比 + 谱质心 + 主频，
  然后用 `cv.py` 量化收益；若无提升，则问题在类别不均衡（CAR 占 60% 窗口）而非特征。

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
# 推荐配置: motion_encoder + 24 维特征 + session 划分 + macro-F1 选模型
python train.py --model motion_encoder --epochs 60 \
    --split_mode session --select_metric macro_f1 --target_hz 50 \
    --drop_features yaw,pitch,roll

# 其余模型 (同口径对比)
python train.py --model cnn_transformer --epochs 60 --split_mode session --select_metric macro_f1
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

### 交叉验证评估（推荐）

```bash
# 按会话分 3 折, 每类会话轮流做验证集, 汇总袋外指标
python cv.py --folds 3 --target_hz 50 --drop_features yaw,pitch,roll
```

评估单个 checkpoint：

```bash
python evaluate.py --checkpoint checkpoints/best_motion_encoder.pth
python export.py --checkpoint checkpoints/best_motion_encoder.pth --onnx
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
├── verify_onnx.py           # ONNX 导出验证 (数值一致性 / 动态 batch)
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

## 移动端部署 (ONNX)

```bash
# 导出 (同时产出 .pth 权重与 .onnx)
python export.py --checkpoint checkpoints/best_motion_encoder.pth --output_dir exported --onnx

# 验证导出结果 (必须跑, 导出成功≠正确)
python verify_onnx.py
```

产物 `exported/motion_motion_encoder.onnx`（约 0.6 MB，单文件，权重内嵌）：

| 属性 | 值 |
|------|-----|
| 输入 | `input`, float32, **`[batch_size, 150, 24]`**（batch 动态） |
| 输出 | `output`, float32, **`[batch_size, 5]`**（batch 动态） |
| opset | 18 |
| 参数量 | 104,006 |

类别顺序固定为：`STATIONARY=0, WALKING=1, CYCLING=2, CAR=3, TRAIN=4`。

输入是**已标准化**的 24 维特征（去 yaw/pitch/roll），不是原始传感器值。
标准化参数 `scaler_mean` / `scaler_scale` 存在 checkpoint 里，
移动端必须用同一套参数做标准化，否则输出无意义：

```python
# 移动端预处理伪代码
x = (raw_features - scaler_mean) / scaler_scale   # raw_features: [150, 24]
probs = session.run(None, {"input": x[None]})[0]  # [1, 5]
```

150 帧 = **3 秒**（采样率固定 50 Hz，重采样逻辑见「数据管线」）。

### 导出验证结果

`verify_onnx.py` 会跑三项检查，当前全部通过：

| 检查 | 结果 |
|------|------|
| `onnx.checker` | 通过（opset 18，598 KB） |
| PyTorch vs ONNX 数值一致性（batch 1/2/3/8/32） | max\|Δ\| = 2.38e-06 |
| 真实数据 22,042 样本预测一致率 | **100.000000%** |

### 一个导出坑（务必留意）

torch 2.x 的 `torch.onnx.export` 默认走 **dynamo 导出器**，但它遇到
`nn.TransformerEncoder` 会把 **batch 维特化成具体值**——无论 `dynamic_shapes`
怎么写，输入输出都会被钉成 `[1, ...]`，batch>1 直接报错。
`export.py` 里强制 `dynamo=False` 走 TorchScript 导出器才能保住动态 batch。
如果你改了模型结构，导出后请务必用 `verify_onnx.py` 复验，不要只看"导出成功"。

## 传感器特征 (27 维)

**基础 (10 维):** gps_speed(km/h), acc_x/y/z, gyro_x/y/z, roll, pitch, yaw

**时域派生 (11 维):** acc/gyro_magnitude, speed_delta,
acc/gyro/roll 滚动均值+标准差, pitch_rolling_mean, vibration_score

**频域派生 (6 维):** acc_bp_0.5_2 / acc_bp_2_5 / acc_bp_5_10 / acc_bp_10_25（各频带能量占比）、
acc_centroid（谱质心）、acc_dom_freq（主频）

> 默认训练用 `--drop_features yaw,pitch,roll` 剔除绝对姿态角，实际输入 **24 维**。
> 姿态角只反映手机朝向，跨会话不泛化（实测两个 STATIONARY 会话的 yaw 差 2.84σ）。

## License

Private research project.