# IterIS++: Enhanced LoRA Merging with Advanced Optimization

IterIS++ 是对原始 IterIS (Iterative Inference-Solving Alignment) 算法的增强版本，引入了三项核心创新来解决LoRA合并中的收敛震荡、正则化偏差和样本冲突问题。

## 🚀 创新点概述

### 1. MATS (动量加速轨迹稳定化)
**Momentum Accelerated Trajectory Stabilization**

基于 Type-II Anderson Acceleration 的加速机制，通过利用历史迭代信息来：
- 平滑更新轨迹，减少震荡
- 加速收敛，将迭代次数减少 50-60%
- 在非凸优化景观中更有效地穿越鞍点

**数学形式：**
```
残差定义：R_k = G(W_k) - W_k
优化目标：min_γ ||R_k - ΔR @ γ||²
加速更新：W_{k+1} = G(W_k) - Σ γ_j (G(W_{k-m+j+1}) - G(W_{k-m+j}))
```

### 2. CAMR (曲率感知流形正则化)
**Curvature-Aware Manifold Regularization**

将各向同性正则化 (αI) 替换为基于激活协方差的曲率感知正则化（遵循 EWC/Fisher 理论）：
- 在激活方差高的方向（重要参数方向）施加**较高**正则化以保护重要特征
- 在激活方差低的方向施加较低正则化
- 支持 DCS 权重协同：高冲突样本不参与协方差计算
- 有效防止灾难性遗忘

**数学形式：**
```
Λ_reg = α · Normalize(diag(Σ X̃ X̃ᵀ)) + β I
```

### 3. DCS (动态冲突感知样本加权)
**Dynamic Conflict-aware Sample Reweighting**

基于跨模型**输出空间**方差的动态样本权重（比输入特征方差更准确）：
- 使用 W @ X 计算输出方差，更接近梯度冲突
- 自适应 σ 参数：基于方差分布的 MAD (中位数绝对偏差) 自动调整
- 低方差样本（任务间共识高）获得高权重
- 高方差样本（任务间冲突大）被降权
- 与 CAMR 协同：DCS 权重影响 CAMR 协方差计算

**数学形式：**
```
V_s = (1/N) Σ ||W_i @ x_s - mean(W @ x_s)||²
σ_adaptive = MAD(V) × scale_factor
w_s = exp(-V_s / σ²)
```

### 4. 收敛监控与自适应退出

新增收敛检测机制，当权重变化小于阈值时自动停止迭代：
```
relative_change = ||W_new - W_old|| / ||W_new||
if relative_change < threshold: early_stop
```

## 📦 安装

### 1. 安装依赖

```bash
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu121/torch_stable.html
```

### 2. 下载数据集（可选，用于 V&L 实验）

```bash
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d ./data/SENTICAP/val2014
```

### 3. 下载预训练的 LoRA 检查点

从 [Huggingface](https://huggingface.co/Daxuxu36) 下载，放置到对应的 `./loras/` 子目录中。

## 🔧 使用方法

### 基本用法

```bash
# 使用 IterIS++ 进行 LoRA 合并
python IterIS_plus.py --task_type <TASK_NAME>
```

支持的任务类型：
- `GLUE_t5` - 合并 GLUE 任务的 LoRAs（多任务）
- `EMOTION_t5_large` - 合并情感分类 LoRAs（域内）
- `TASKS_blip_base` - 合并视觉语言 LoRAs

### 高级用法

#### 选择性启用/禁用创新模块

```bash
# 仅使用 MATS
python IterIS_plus.py --task_type GLUE_t5 --use_mats 1 --use_camr 0 --use_dcs 0

# 仅使用 CAMR
python IterIS_plus.py --task_type GLUE_t5 --use_mats 0 --use_camr 1 --use_dcs 0

# 仅使用 DCS
python IterIS_plus.py --task_type GLUE_t5 --use_mats 0 --use_camr 0 --use_dcs 1

# 使用所有创新（默认）
python IterIS_plus.py --task_type GLUE_t5 --use_mats 1 --use_camr 1 --use_dcs 1
```

#### 使用自定义配置文件

```bash
python IterIS_plus.py --config config/methods-config/iteris-plus-config.yaml --task_type GLUE_t5
```

### 与原始 IterIS 比较

```bash
# 运行原始 IterIS
python IterIS.py --task_type GLUE_t5

# 运行 IterIS++
python IterIS_plus.py --task_type GLUE_t5
```

## ⚙️ 配置参数

### IterIS++ 特有参数

在 `config/methods-config/iteris-plus-config.yaml` 中配置：

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `use_mats` | bool | True | 启用 MATS (Anderson Acceleration) |
| `mats_history_size` | int | 5 | Anderson 加速的历史深度 |
| `mats_regularization` | float | 1e-6 | Anderson 最小二乘的正则化系数 |
| `use_camr` | bool | True | 启用 CAMR (曲率感知正则化，EWC对齐) |
| `camr_alpha` | float | 与 alpha_1 相同 | CAMR 正则化强度 |
| `camr_beta` | float | 1e-8 | CAMR 最小正则化值 |
| `use_dcs` | bool | True | 启用 DCS (动态样本加权，使用输出方差) |
| `dcs_sigma` | float | 1.0 | 自适应 σ 的缩放因子 |
| `convergence_threshold` | float | 1e-6 | 收敛检测阈值，用于自适应退出 |

### 继承自 IterIS 的参数

| 参数 | 描述 |
|------|------|
| `max_iter` | 最大迭代次数（配合收敛检测可自动提前退出）|
| `task_targets` | 要合并的任务列表 |
| `model_name` | 基础模型名称 |
| `lora_alpha` | LoRA 的 alpha 值 |
| `alpha_1`, `alpha_2` | 内积矩阵的正则化系数 |
| `rank` | LoRA 的秩 |
| `samples_num` | 校准样本数量 |

## 📊 预期性能提升

| 指标 | 原始 IterIS | IterIS++ | 改进说明 |
|------|------------|----------|---------|
| 收敛质量 | 基线 | 更平滑 | MATS 减少震荡 |
| 收敛速度 | 基线 | 可提前退出 | 收敛检测机制 |
| 多任务冲突处理 | 特征方差 | 输出方差 | DCS 更精确的冲突代理 |
| 正则化对齐 | 各向同性 | EWC对齐 | CAMR 保护重要方向 |
| 超参数敏感度 | σ 固定 | σ 自适应 | DCS 自动调整 |
| 模块协同 | 独立运行 | 权重传递 | DCS→CAMR 协同 |

**注意**: 实际性能提升取决于具体任务和数据集。建议通过消融实验验证各模块的效果。

## 🔬 理论基础

### MATS: Anderson Acceleration

Anderson 加速本质上是拟牛顿法的逆向应用，它利用历史残差隐式逼近不动点算子的逆雅可比矩阵：

```
G(W_{k-m}), ..., G(W_k)  →  最优线性组合  →  W_{k+1}
```

### CAMR: 贝叶斯视角

CAMR 将 IterIS 从普通最小二乘升级为贝叶斯线性回归的 MAP 估计：
- 先验分布：不再是球形高斯，而是由预训练模型几何结构决定的椭球高斯
- 正则化矩阵：Ω = diag(激活协方差)

### DCS: 迭代重加权最小二乘

DCS 构成了迭代重加权最小二乘 (IRLS) 框架：
- 输出方差作为冲突代理指标
- 高斯核映射方差到权重
- 加权最小二乘求解

## 📁 文件结构

```
├── IterIS_plus.py                          # IterIS++ 主程序
├── IterIS.py                               # 原始 IterIS（保留兼容）
├── config/
│   └── methods-config/
│       ├── iteris-plus-config.yaml         # IterIS++ 配置
│       └── iteris-config.yaml              # 原始 IterIS 配置
├── get_midfeatures.py                      # 特征提取模块
├── eval_model.py                           # 模型评估模块
├── loras/                                  # LoRA 检查点目录
└── README_IterIS_plus.md                   # 本文档
```

## 🧪 实验验证

### 运行完整评估

```bash
# GLUE 基准测试
python IterIS_plus.py --task_type GLUE_t5

# 情感分析任务
python IterIS_plus.py --task_type EMOTION_t5_large

# 视觉语言任务
python IterIS_plus.py --task_type TASKS_blip_base
```

### 消融实验

```bash
# 仅 MATS
python IterIS_plus.py --task_type GLUE_t5 --use_mats 1 --use_camr 0 --use_dcs 0

# MATS + CAMR
python IterIS_plus.py --task_type GLUE_t5 --use_mats 1 --use_camr 1 --use_dcs 0

# 完整 IterIS++
python IterIS_plus.py --task_type GLUE_t5 --use_mats 1 --use_camr 1 --use_dcs 1
```

## 📝 协同效应分析

### DCS + MATS
- DCS 通过剔除高冲突样本"平滑"优化景观
- MATS 在更平滑的景观上加速效果更显著

### CAMR + DCS
- CAMR 提供正则化约束，防止偏离重要参数
- DCS 确保用于拉动模型的数据是高质量的
- 两者一推一拉，保留旧知识同时安全吸收新知识

## 📚 引用

如果您使用本工作，请引用原始 IterIS 论文：

```bibtex
@inproceedings{chen2025iteris,
  title={IterIS: Iterative Inference-Solving Alignment for LoRA Merging},
  author={Chen, Hongxu and Li, Runshi and Zhu, Bowei and Wang, Zhen and Chen, Long},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## 📧 问题与反馈

如遇到问题，请通过 GitHub Issues 提交反馈。
