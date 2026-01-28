# 脚本系统更新总结

**日期**: 2026-01-28

## 🎉 更新内容

已创建全新的三阶段脚本系统，取代原有的四阶段手动流程。

## 📁 新增文件

### 核心脚本

1. **stage1_data_generation.py** (7.1K)
   - 数据生成阶段
   - 自动配置、运行、保存日志、拷贝数据

2. **stage2_model_training.py** (11K)
   - 模型训练阶段
   - 支持编码器+GAN训练，或仅编码器训练
   - 支持批量训练多个方法

3. **stage3_inference_testing.py** (8.6K)
   - 推理测试阶段
   - 支持批量测试多个方法
   - 支持测试所有方法、所有GAN方法、所有传统方法

4. **run_experiment.py** (10K)
   - 统一实验运行脚本
   - 可选择运行的阶段和方法
   - 一键运行完整实验流程

### 文档

1. **README_NEW.md**
   - 完整的使用文档
   - 包含所有功能说明、使用示例、常见场景

2. **QUICKSTART.md**
   - 快速开始指南
   - 三种使用方式
   - 常见场景和FAQ

## ✨ 核心特性

### 1. 三阶段流程

合并原有的Stage2和Stage3：
- **阶段1**：数据生成（1000步，无恢复机制）
- **阶段2**：模型训练（编码器自动训练 + GAN在线训练）
- **阶段3**：推理测试（100步，training=False）

### 2. 自动化运行

- ✅ 自动配置 main.py
- ✅ 自动运行 main.py
- ✅ 实时显示输出
- ✅ 自动保存日志
- ✅ 自动拷贝数据（阶段1）

### 3. 灵活配置

```bash
# 指定方法
--method PreGAN
--methods PreGAN PreGANPlus

# 指定参数
--steps 1200
--new-containers 5
--log-dir experiment_logs/

# 批量选择
--all              # 所有方法
--all-gan          # 所有GAN方法
--all-traditional  # 所有传统方法
```

### 4. 编码器优化模式

新增 `--encoder-only` 参数：
- 只训练编码器，不训练GAN
- 适用于故障检测准确率优化
- 使用短步数（10步）触发训练

### 5. 批量操作

```bash
# 批量训练
python3 scripts/stage2_model_training.py --methods PreGAN PreGANPlus

# 批量测试
python3 scripts/stage3_inference_testing.py --all
```

## 🔧 训练机制确认

经过代码分析确认：

### 编码器训练
- **触发条件**：`epoch == -1`（checkpoint不存在或未训练）
- **训练时机**：运行时自动触发
- **训练数据**：阶段1生成的数据（离线训练）
- **数据路径**：`recovery/PreGANSrc/data/simulator/`

### GAN训练
- **训练方式**：在线训练（使用运行时数据）
- **训练参数**：`training=True` 时才训练
- **训练步数**：由 `NUM_SIM_STEPS` 控制（默认1200）

### 训练流程
1. 加载编码器checkpoint
2. 如果 `epoch == -1`，自动训练编码器（使用阶段1数据）
3. 编码器训练完成后冻结
4. 如果 `training=True`，继续进行GAN在线训练
5. GAN使用当前运行时的新数据进行训练

## 📊 与旧脚本对比

| 特性 | 旧脚本 | 新脚本 |
|------|-------|-------|
| 阶段数 | 4个 | 3个（合并2+3） |
| 自动运行main.py | ❌ | ✅ |
| 自动保存日志 | ❌ | ✅ |
| 自动拷贝数据 | ❌ | ✅ |
| 批量操作 | ❌ | ✅ |
| 编码器优化 | ❌ | ✅ |
| 统一入口 | ⚠️ (bash) | ✅ (python) |
| 参数化配置 | ⚠️ | ✅ |

## 🎯 使用建议

### 推荐使用方式

**方式1：统一脚本**（最简单）
```bash
python3 scripts/run_experiment.py --all-stages --methods PreGAN
```

**方式2：分阶段脚本**（最灵活）
```bash
python3 scripts/stage1_data_generation.py
python3 scripts/stage2_model_training.py --method PreGAN
python3 scripts/stage3_inference_testing.py --method PreGAN
```

### 常见场景

1. **首次运行**：使用统一脚本
2. **编码器优化**：使用 `--encoder-only`
3. **方法比较**：使用批量测试
4. **继续训练**：直接重新运行阶段2

## 🔄 迁移指南

### 从旧脚本迁移

**旧方式**：
```bash
# 需要多次手动操作
python3 scripts/paper_experiment_stage1_data_collection.py
python3 main.py -e "" -m 0
# 手动检查日志...
# 手动拷贝数据...

python3 scripts/paper_experiment_stage2_encoder_training.py --method PreGAN
python3 main.py -e "" -m 0
# 手动保存日志...

python3 scripts/paper_experiment_stage3_gan_training.py --method PreGAN
python3 main.py -e "" -m 0

python3 scripts/paper_experiment_stage4_testing.py --method PreGAN
python3 main.py -e "" -m 0
```

**新方式**：
```bash
# 一条命令完成
python3 scripts/run_experiment.py --all-stages --methods PreGAN

# 或分步运行（更灵活）
python3 scripts/stage1_data_generation.py
python3 scripts/stage2_model_training.py --method PreGAN
python3 scripts/stage3_inference_testing.py --method PreGAN
```

## 📝 注意事项

1. **旧脚本保留**：旧脚本仍然可用，但建议使用新脚本
2. **数据兼容**：新旧脚本生成的数据和模型完全兼容
3. **配置方式**：两种方式都是修改 main.py，机制相同
4. **日志位置**：新脚本使用独立的日志目录（可配置）

## 🐛 已知问题

暂无

## 🚀 未来改进

可能的改进方向：
1. 支持配置文件（YAML/JSON）
2. 实验结果自动分析和可视化
3. 支持并行训练多个方法
4. 集成到 CI/CD 流程

## 📞 支持

如有问题，请查看：
- [README_NEW.md](README_NEW.md) - 完整文档
- [QUICKSTART.md](QUICKSTART.md) - 快速开始
- [旧版README.md](README.md) - 旧脚本文档

---

**总结**：新脚本系统提供了更自动化、更灵活、更易用的实验流程管理。建议在新的实验中使用新脚本系统。
