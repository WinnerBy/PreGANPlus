# 阶段4测试脚本使用说明

## 概述

这两个脚本用于运行所有方法的阶段4测试，并将每个方法的结果保存到独立目录，便于对比分析。

## 脚本说明

### 1. `run_stage4_all_methods.py`
**功能**：运行所有7个方法的测试（PreGAN, PreGANPlus, PreGANPlusEnhanced, PCFT, DFTM, ECLB, CMODLB）

**工作流程**：
1. 对每个方法：
   - 配置main.py用于该方法的测试
   - 运行100步测试
   - 将结果备份到 `logs/stage4_results/{method_name}/` 目录
2. 汇总所有方法的运行状态

**使用方法**：
```bash
python3 scripts/run_stage4_all_methods.py
```

**输出**：
- 每个方法的结果保存在：`logs/stage4_results/{method_name}/RPiEdge_BWGD2_100_.../`
- 所有结果互不覆盖，可以同时保存

### 2. `analyze_stage4_results.py`
**功能**：分析所有方法的测试结果并生成对比报告

**分析内容**：
- 总能量 (Total Energy)
- 平均响应时间 (Average Response Time)
- 迁移次数 (Migration Count)
- SLA违约数 (SLA Violations)
- SLA违约率 (SLA Violation Rate)

**使用方法**：
```bash
python3 scripts/analyze_stage4_results.py
```

**输出**：
- 控制台显示对比表格和详细分析
- 报告文件：`logs/stage4_results/comparison_report.txt`

## 完整流程

```bash
# 步骤1：运行所有方法的测试（这可能需要较长时间）
python3 scripts/run_stage4_all_methods.py

# 步骤2：分析所有结果
python3 scripts/analyze_stage4_results.py
```

## 注意事项

1. **运行时间**：每个方法需要运行100步测试，7个方法总共可能需要较长时间
2. **结果保存**：每个方法的结果会保存到独立目录，不会相互覆盖
3. **已训练模型**：确保所有方法的模型都已训练完成（阶段2和阶段3）
4. **内存使用**：如果内存不足，可以分批运行（修改脚本中的METHODS列表）

## 结果目录结构

```
logs/
└── stage4_results/
    ├── PreGAN/
    │   └── RPiEdge_BWGD2_100_.../
    ├── PreGANPlus/
    │   └── RPiEdge_BWGD2_100_.../
    ├── PreGANPlusEnhanced/
    │   └── RPiEdge_BWGD2_100_.../
    ├── PCFT/
    │   └── RPiEdge_BWGD2_100_.../
    ├── DFTM/
    │   └── RPiEdge_BWGD2_100_.../
    ├── ECLB/
    │   └── RPiEdge_BWGD2_100_.../
    ├── CMODLB/
    │   └── RPiEdge_BWGD2_100_.../
    └── comparison_report.txt
```

