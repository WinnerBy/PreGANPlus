# PreDiffusion 设计笔记

## 概述
本笔记记录将 PreGANPlus 中的 GAN 替换为条件扩散模型（Diffusion Model）的完整设计与实施计划。新方案命名为 PreDiffusion（或 PreGAN-Diffusion），目标是保持现有 encoder（Transformer）结构，替换生成器以提升生成稳定性与质量，同时与原 `PreGAN` / `PreGANPlus` 做可比实验。

---

## 目标
- 用条件 DDPM/DDIM（扩散模型）替换原有 GAN，生成改进后的调度（schedule）。
- 保留或复用 PreGANPlus 的 encoder（Transformer）作为 condition 输入。 
- 实现独立的 `Recovery` 子类，命名为 `PreDiffusionRecovery`，接口与 `PreGANPlusRecovery` 兼容，便于实验框架替换。

---

## 目录与新增文件
- `recovery/PreDiffusion.py`  
  - 类：`PreDiffusionRecovery(Recovery)`
  - 功能：加载 encoder、加载/训练 diffusion、run_encoder、run_model、recover_decision

- `recovery/PreGANSrc/src/diffusion.py`  
  - 内容：`DiffusionModel`（denoiser）、`train_diffusion`、`sample_diffusion`、`save_diffusion`、`load_diffusion`。

- `recovery/PreGANSrc/src/utils.py`（可选）  
  - 新增：`save_diffusion` / `load_diffusion` helpers、evaluation helper（若需）

- 实验脚本（可选）
  - `scripts/batch_run_experiments.py`：加入 `--recovery PreDiffusion` 选项

---

## 模型设计要点
### 条件方式
- 起步采用简单 concat：将 encoder 对每个容器/主机的 embedding 与 schedule 行向量 concat 作为 denoiser 的条件输入；全局 embedding 可扩展为每行的附加特征。
- 后续可换成 FiLM 或 cross-attention 注入。

### Denoiser 架构
- 最小可行：按行的 MLP denoiser（per-container MLP），输入维度 = noise_row + condition_dim + t_embed。
- 进阶可选：小型 UNet 或 Transformer-UNet，处理整个 schedule 矩阵（更强但复杂）。

### 训练与采样
- 训练：标准 DDPM 训练（预测噪声 epsilon，MSE loss），beta schedule（linear 或 cosine），T=1000（训练）；optimizer AdamW lr=1e-4。
- 采样：支持 DDIM 少步采样以加速（尝试 10/25/50 步）。

---

## API 设计（函数接口示例）
- `DiffusionModel.sample(condition: Tensor, schedule: Tensor, steps: int) -> Tensor`  # 返回 new_schedule_data
- `DiffusionModel.train_step(condition: Tensor, schedule: Tensor) -> loss`  # single training step
- `save_diffusion(path, model_state, optimizer_state)` / `load_diffusion(path)`

`PreDiffusionRecovery` 中调用点：
- `run_encoder(schedule_data)`（复用现有 transformer）
- `train_diffusion(embedding, schedule_data)`（在 training 模式或在线调优时调用）
- `sample_diffusion(embedding, schedule_data, steps)`（在 recover_decision 前调用）

---

## 评估方案（与现有对比）
- 保持原有评估：`run_simulation(new_schedule)`, `run_simulation(original_schedule)`。
- 收集指标：迁移后系统得分差、迁移次数、SLA 违约率、迁移带宽、采样耗时、检测准确率。
- 消融：采样步数敏感性（10/25/50），条件注入方法（concat vs FiLM vs attention），与原 GAN/无GAN/只encoder 的对比。

---

## 实施步骤与时间估计
1. 最小实现（MLP denoiser + sample/train API）——2–4 天
2. 集成 `PreDiffusionRecovery` 并做 smoke test（单样本）——1–2 天
3. 加入 DDIM 少步采样与优化——2–3 天
4. 完整对比实验（多次重复）与数据收集——1–2 周

---

## 风险与缓解
- 训练样本不足：用现有 GAN 生成 warm-start 样本或用原 schedule 做自监督训练。
- 采样慢：优先使用 DDIM 少步采样或在高风险时才采样较多步（adaptive）。
- 实现复杂：分阶段验证（先 MLP，再 UNet）。

---

## 下一步（建议）
- 确认文件名/类名（默认：`PreDiffusionRecovery` 与 `DiffusionModel`），我将实现最小可行模块并在 `PreGANPlus` 流程里做接口对接与 smoke test。

---

如需我现在开始实现最小版（我会创建 `diffusion.py` 与 `PreDiffusion.py` 并运行 smoke test），直接回复确认即可。