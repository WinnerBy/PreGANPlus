#!/bin/bash
# 按照论文流程运行完整实验
# 阶段1：数据收集 -> 阶段2+3：编码器训练（自动）+ GAN训练 -> 阶段2：CMODLB编码器训练 -> 阶段4：测试评估

set -e

# 激活conda环境
# source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null
# conda activate pregan

# 进入项目根目录
cd "$(dirname "$0")/.."

echo "=========================================="
echo "论文实验流程"
echo "=========================================="
echo ""

# 创建实验日志目录
LOG_DIR="experiment_logs/paper_experiment"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 阶段1：数据收集
echo "=========================================="
echo "阶段1：数据收集（1000步，无恢复）"
echo "=========================================="
echo "📝 注意："
echo "   - 此阶段生成1000步数据用于后续编码器训练"
echo "   - 如果1000步数据训练效果不好，可以继续生成更多数据"
echo "   - 生成的数据会自动保存到 logs/ 目录，需手动确认使用哪个数据集"
echo ""
python3 scripts/paper_experiment_stage1_data_collection.py
python3 main.py -e "" -m 0 2>&1 | tee "$LOG_DIR/stage1_data_collection_${TIMESTAMP}.log"
echo "✅ 阶段1完成"
echo ""

# 数据管理：找到最新生成的数据并拷贝到训练目录
echo "数据管理：查找并拷贝阶段1生成的数据..."
LATEST_LOG_DIR=$(find logs -maxdepth 1 -name "RPiEdge_BWGD2_1000*" -type d | sort -r | head -1)
if [ -z "$LATEST_LOG_DIR" ]; then
    echo "❌ 错误：未找到阶段1生成的数据（RPiEdge_BWGD2_1000*）"
    echo "   请确认阶段1运行成功，NUM_SIM_STEPS=1000，且数据已保存到 logs/ 目录"
    exit 1
fi
echo "✅ 找到数据目录：$LATEST_LOG_DIR"

# 确保目标目录存在
mkdir -p "recovery/PreGANSrc/data/simulator"

# 拷贝数据到训练数据目录
cp "$LATEST_LOG_DIR/time_series.npy" "recovery/PreGANSrc/data/simulator/time_series.npy"
cp "$LATEST_LOG_DIR/schedule_series.npy" "recovery/PreGANSrc/data/simulator/schedule_series.npy"
echo "✅ 数据已拷贝到 recovery/PreGANSrc/data/simulator/"
echo ""

# 阶段2+3合并：编码器训练 + GAN训练（PreGAN/PreGANPlus/PreGANPlusEnhanced）
echo "=========================================="
echo "阶段2+3合并：编码器训练 + GAN训练（1200步）"
echo "=========================================="
echo "📝 注意："
echo "   - 编码器训练是自动的：如果checkpoint不存在或epoch == -1，会自动训练"
echo "   - 编码器训练使用阶段1收集的1000步数据（离线训练）"
echo "   - 编码器训练完成后，会继续运行1200步的GAN训练"
echo "   - PreGAN和PreGANPlus使用不同的编码器（FPE vs Transformer）"
echo "   - PreGANPlus和PreGANPlusEnhanced共享相同的Transformer编码器"
echo "   - Transformer编码器只在PreGANPlus阶段训练一次，PreGANPlusEnhanced直接使用"
echo "   - 如果效果不好，可以多训练几次"
echo "   - 如果程序被终止，可以从已有checkpoint继续训练"
echo ""

# 先训练PreGAN（FPE编码器）
echo ""
echo "--- 训练 PreGAN（FPE编码器自动训练 + GAN训练）---"
python3 scripts/paper_experiment_stage3_gan_training.py --method PreGAN
python3 main.py -e "" -m 0 2>&1 | tee "$LOG_DIR/stage2+3_training_PreGAN_${TIMESTAMP}.log"
echo "✅ PreGAN 训练完成"

# 然后训练PreGANPlus（Transformer编码器，会训练并保存）
echo ""
echo "--- 训练 PreGANPlus（Transformer编码器自动训练 + GAN训练）---"
python3 scripts/paper_experiment_stage3_gan_training.py --method PreGANPlus
python3 main.py -e "" -m 0 2>&1 | tee "$LOG_DIR/stage2+3_training_PreGANPlus_${TIMESTAMP}.log"
echo "✅ PreGANPlus 训练完成（Transformer编码器已保存）"

# 最后训练PreGANPlusEnhanced（使用已训练的Transformer编码器，不重复训练）
echo ""
echo "--- 训练 PreGANPlusEnhanced（使用已训练的Transformer编码器 + GAN训练）---"
echo "📝 注意：PreGANPlusEnhanced使用与PreGANPlus相同的Transformer编码器"
echo "   - Transformer编码器已在PreGANPlus阶段训练完成"
echo "   - 此阶段直接加载已训练的Transformer，不重复训练"
python3 scripts/paper_experiment_stage3_gan_training.py --method PreGANPlusEnhanced
python3 main.py -e "" -m 0 2>&1 | tee "$LOG_DIR/stage2+3_training_PreGANPlusEnhanced_${TIMESTAMP}.log"
echo "✅ PreGANPlusEnhanced 训练完成"

echo ""
echo "✅ 阶段2+3完成"
echo ""

# 阶段2：CMODLB编码器训练（单独处理，因为它不需要GAN训练）
echo "=========================================="
echo "阶段2：CMODLB编码器训练（1200步）"
echo "=========================================="
echo "📝 注意：CMODLB不需要GAN训练，只需要编码器训练"
echo "   - 编码器的训练是自动进行的"
echo "   - 如果checkpoint不存在，会在首次运行时自动训练"
echo "   - 训练数据来自 recovery/PreGANSrc/data/"
echo "   - 训练完成后模型保存到 checkpoints/ 目录"
echo "   - 使用1200步确保环境稳定，统计数据准确"
echo ""

for method in CMODLB; do
    echo ""
    echo "--- 训练 $method 编码器 ---"
    python3 scripts/paper_experiment_stage2_encoder_training.py --method "$method"
    python3 main.py -e "" -m 0 2>&1 | tee "$LOG_DIR/stage2_encoder_training_${method}_${TIMESTAMP}.log"
    echo "✅ $method 编码器训练完成"
done

echo ""
echo "✅ 阶段2（CMODLB）完成"
echo ""

# 阶段4：测试评估（所有方法对比）
echo "=========================================="
echo "阶段4：测试评估（100步，所有方法对比）"
echo "=========================================="

# 测试所有方法
for method in PreGAN PreGANPlus PreGANPlusEnhanced PCFT DFTM ECLB CMODLB; do
    echo ""
    echo "--- 测试 $method ---"
    python3 scripts/paper_experiment_stage4_testing.py --method "$method"
    python3 main.py -e "" -m 0 2>&1 | tee "$LOG_DIR/stage4_testing_${method}_${TIMESTAMP}.log"
    echo "✅ $method 测试完成"
done

echo ""
echo "=========================================="
echo "✅ 所有阶段完成！"
echo "=========================================="
echo ""
echo "实验结果保存在："
echo "  - 日志：$LOG_DIR/"
echo "  - 数据：logs/"
echo "  - 模型：recovery/PreGANSrc/checkpoints/"
echo ""

