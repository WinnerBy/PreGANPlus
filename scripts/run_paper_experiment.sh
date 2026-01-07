#!/bin/bash
# 按照论文流程运行完整实验
# 阶段1：数据收集 -> 阶段2：FPE训练（自动）-> 阶段3：GAN训练 -> 阶段4：测试评估

set -e

# 激活conda环境
source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null
conda activate pregan

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
echo "阶段1：数据收集（500步，无恢复）"
echo "=========================================="
echo "📝 注意：如果500步数据训练效果不好，可以继续生成更多数据"
echo ""
python3 scripts/paper_experiment_stage1_data_collection.py
python3 main.py -e "" -m 0 2>&1 | tee "$LOG_DIR/stage1_data_collection_${TIMESTAMP}.log"
echo "✅ 阶段1完成"
echo ""

# 阶段2：编码器训练（FPE/Transformer/FCN）
echo "=========================================="
echo "阶段2：编码器训练（FPE/Transformer/FCN）"
echo "=========================================="
echo "📝 注意：编码器的训练是自动进行的"
echo "   - 如果checkpoint不存在，会在首次运行时自动训练"
echo "   - 训练数据来自 recovery/PreGANSrc/data/"
echo "   - 训练完成后模型保存到 checkpoints/ 目录"
echo ""

# 训练所有需要编码器的方法
for method in PreGAN PreGANPlus PreGANPlusEnhanced CMODLB; do
    echo ""
    echo "--- 训练 $method 编码器 ---"
    python3 scripts/paper_experiment_stage2_encoder_training.py --method "$method"
    python3 main.py -e "" -m 0 2>&1 | tee "$LOG_DIR/stage2_encoder_training_${method}_${TIMESTAMP}.log"
    echo "✅ $method 编码器训练完成"
done

echo ""
echo "✅ 阶段2完成"
echo ""

# 阶段3：GAN训练（每个方法分别训练）
echo "=========================================="
echo "阶段3：在线训练GAN（300步，可根据需要调整）"
echo "=========================================="
echo "📝 注意："
echo "   - 如果效果不好，可以多训练几次"
echo "   - 如果程序被终止，可以从已有checkpoint继续训练"
echo "   - 中间保存频率：每50步保存一次stats（减少内存压力）"
echo ""

for method in PreGAN PreGANPlus PreGANPlusEnhanced; do
    echo ""
    echo "--- 训练 $method ---"
    python3 scripts/paper_experiment_stage3_gan_training.py --method "$method"
    python3 main.py -e "" -m 0 2>&1 | tee "$LOG_DIR/stage3_gan_training_${method}_${TIMESTAMP}.log"
    echo "✅ $method 训练完成"
done

echo ""
echo "✅ 阶段3完成"
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

