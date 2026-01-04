#!/bin/bash
# 运行完整对比实验脚本
# 运行FPE-GAN, TF-GAN, MAMO-GAN的对比实验
#
# 方法命名:
#   - FPE-GAN (Fault Prediction Encoder GAN): 原PreGAN方法
#   - TF-GAN (Transformer-based Fault GAN): 原PreGANPlus方法
#   - MAMO-GAN (Migration-Aware Multi-Objective GAN): 我们的改进方法

# 激活conda环境
source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null
conda activate pregan

# 创建日志目录
LOG_DIR="experiment_logs"
mkdir -p $LOG_DIR

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/experiment_${TIMESTAMP}.log"

echo "=========================================="
echo "运行完整对比实验"
echo "方法: FPE-GAN, TF-GAN, MAMO-GAN"
echo "时间: $(date)"
echo "日志文件: $LOG_FILE"
echo "=========================================="
echo ""

# 运行实验，同时输出到终端和文件
python scripts/batch_run_experiments.py \
    --models PreGAN,PreGANPlus,PreGANPlusEnhanced \
    --steps 100 \
    2>&1 | tee $LOG_FILE

# 检查退出状态
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "实验完成！"
    echo "日志文件: $LOG_FILE"
    echo "结果文件: all_datasets/simulator/"
    echo "图表文件: results/simulator/"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "实验失败，请查看日志: $LOG_FILE"
    echo "=========================================="
    exit 1
fi

