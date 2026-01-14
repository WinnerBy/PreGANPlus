# 实验结果分析

**创建日期**: 2026-01-14  
**实验日期**: 2026-01-12 至 2026-01-13  
**总运行次数**: 116次

---

## 📚 文档导航

本目录包含实验结果的详细分析：

1. **[对比分析](Comparative_Analysis.md)** - 方法间的全面对比
2. **[性能指标分析](Performance_Analysis.md)** - 详细的性能指标分析
3. **[详细发现](Detailed_Findings.md)** - 深入的发现和案例研究

---

## 📊 结果文件位置

所有结果文件保存在 `final_results/` 目录：

- **数据**: `final_results/data/` - 最终选中的实验数据
- **日志**: `final_results/logs/` - 最终选中的日志文件
- **汇总**: `final_results/summary/` - 汇总数据和报告
- **图表**: `final_results/plots/` - 最终对比图表

---

## 🎯 关键发现摘要

### MAMO-GAN vs TF-GAN ✅

- 能耗降低: **-1.18%**
- 响应时间改善: **-8.65%**
- SLA违规减少: **-15.93%**
- 迁移数增加: +10.19% (可接受)

### FPE-GAN vs 传统方法 ✅

- 迁移数减少: **21%-84%**
- 响应时间优于大部分传统方法
- 能耗与传统方法相当或略优

---

## 📈 对比图表

详细的对比图表请参考：
- `final_results/plots/group1_pregán_vs_traditional/` - FPE-GAN vs 传统方法
- `final_results/plots/group2_pregánplus_vs_pregán/` - TF-GAN vs FPE-GAN
- `final_results/plots/group3_pregánplusenhanced_vs_others/` - MAMO-GAN vs Others

---

## 🔗 相关文档

- [方法设计文档](../01_Methods/README.md) - 三种GAN方法的详细设计
- [实验设计文档](../02_Experiments/README.md) - 实验配置和流程
- [最终结果报告](../../final_results/summary/最终选择结果报告.md) - 完整的结果报告

---

**最后更新**: 2026-01-14
