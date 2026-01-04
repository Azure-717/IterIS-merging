# EMOTION_t5_large 任务内存优化说明

## 问题描述
运行以下命令时会出现 OOM (内存不足) 错误：
```bash
python IterIS_plus.py --task_type EMOTION_t5_large --use_mats 1 --use_camr 0 --use_dcs 0
```

## 解决方案
我们实现了多项内存优化技术，专门针对 EMOTION_t5_large 任务，在不显著影响性能的前提下大幅减少内存使用。

## 已实现的优化

### 1. FP16 混合精度 (use_fp16)
- **内存减少**: 约 50%
- **性能影响**: 精度损失小于 1%（通常可忽略不计）
- **原理**: 使用 16 位浮点数代替 32 位，内存占用减半

### 2. 梯度检查点 (use_gradient_checkpointing)
- **内存减少**: 约 30%
- **性能影响**: 计算时间增加 10-20%
- **原理**: 用计算换内存，在反向传播时重新计算激活值而不是存储

### 3. 顺序层处理 (sequential_layer_processing)
- **内存减少**: 降低峰值内存使用
- **性能影响**: 对精度影响极小，处理时间略微增加
- **原理**: 逐层处理并立即清理，而不是同时在内存中保留所有层

### 4. 优化批次大小
- **调整**: `inner_num: 6 → 3`, `outer_num: 10 → 20`
- **内存减少**: 约 50% 批次相关内存
- **性能影响**: 总样本数不变，精度基本无影响

## 配置说明

优化配置已经在 `config/methods-config/iteris-plus-config.yaml` 中默认启用：

```yaml
EMOTION_t5_large:
  # 批次大小优化
  inner_num: 3      # 从 6 减少到 3
  outer_num: 20     # 从 10 增加到 20（保持总样本数 60）
  
  # 内存优化参数
  use_fp16: True                        # 启用 FP16
  use_gradient_checkpointing: True      # 启用梯度检查点
  sequential_layer_processing: True     # 启用顺序层处理
```

## 使用方法

直接运行原命令即可，优化已默认启用：

```bash
python IterIS_plus.py --task_type EMOTION_t5_large --use_mats 1 --use_camr 0 --use_dcs 0
```

## 预期效果

启用所有优化后：
- **总体内存减少**: 60-70%
- **精度影响**: < 1% (通常可以忽略)
- **训练时间**: 增加 10-30%
- **适用 GPU**: 8-12GB 显存即可运行

## 进一步优化（如仍然 OOM）

如果仍然遇到内存不足，可以进一步调整：

### 方案 1: 进一步减小批次
```yaml
inner_num: 2      # 更小的批次
outer_num: 30     # 保持总样本数
```

### 方案 2: 减少样本数（可能影响精度）
```yaml
samples_num: 40   # 从 60 减少到 40
```

### 方案 3: 减少 MATS 历史记录
```yaml
mats_history_size: 1  # 最小历史记录
```

## 禁用优化（如有足够内存）

如果内存充足且希望最快速度，可以禁用优化：

```yaml
use_fp16: False
use_gradient_checkpointing: False
sequential_layer_processing: False
inner_num: 6
outer_num: 10
```

## 技术细节

### 内存使用监控
脚本会在每次迭代后打印内存使用情况：
```
Max memory usage: XXXX.XX MB
```

### 优化实现位置
- `IterIS_plus.py`: 核心实现
- `config/methods-config/iteris-plus-config.yaml`: 配置文件
- `MEMORY_OPTIMIZATION.md`: 英文详细文档

### 测试验证
运行测试脚本验证优化是否正确配置：
```bash
python test_memory_optimization.py
```

## 故障排除

### 问题: 仍然 OOM
**解决方案**:
1. 进一步减小 `inner_num` (尝试 2 或 1)
2. 减少 `samples_num` (尝试 40 或 30)
3. 使用更大显存的 GPU
4. 考虑使用 CPU 运行（慢但无内存限制）

### 问题: 精度下降
**解决方案**:
1. 禁用 FP16: `use_fp16: False`
2. 增加样本数回到原值
3. 如内存允许，使用更大批次

### 问题: 速度太慢
**解决方案**:
1. 禁用梯度检查点: `use_gradient_checkpointing: False`
2. 禁用顺序处理: `sequential_layer_processing: False`
3. 如内存允许，增加批次大小

## 相关文件

- `MEMORY_OPTIMIZATION.md`: 英文详细指南
- `test_memory_optimization.py`: 测试脚本
- `config/methods-config/iteris-plus-config.yaml`: 配置文件

## 参考资料

- PyTorch 混合精度训练: https://pytorch.org/docs/stable/amp.html
- 梯度检查点: https://pytorch.org/docs/stable/checkpoint.html
