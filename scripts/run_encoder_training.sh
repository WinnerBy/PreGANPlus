#!/bin/bash
# 编码器训练脚本
# 用途：运行编码器训练并保存日志

# 设置日志文件路径
LOG_DIR="experiment_logs/encoder_training"
LOG_FILE="${LOG_DIR}/encoder_training_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "开始编码器训练"
echo "日志文件: ${LOG_FILE}"
echo "=========================================="
echo ""

# 运行训练并保存日志
# 使用tee同时输出到终端和文件
python3 main.py -e "" -m 0 2>&1 | tee "${LOG_FILE}"

echo ""
echo "=========================================="
echo "训练完成"
echo "日志已保存到: ${LOG_FILE}"
echo "=========================================="
