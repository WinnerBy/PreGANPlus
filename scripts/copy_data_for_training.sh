#!/bin/bash
# 复制阶段1生成的数据到训练目录
# 包括：time_series.npy, schedule_series.npy, fault_history.pkl

set -e

# 进入项目根目录
cd "$(dirname "$0")/.."

# 数据源目录（阶段1生成的数据）
DATA_SOURCE_DIR="logs/RPiEdge_BWGD2_1000_16_16_1000_10000_300_5"

# 目标目录（训练数据目录）
TARGET_DIR="recovery/PreGANSrc/data/simulator"

# 检查源目录是否存在
if [ ! -d "$DATA_SOURCE_DIR" ]; then
    echo "❌ 错误：数据源目录不存在：$DATA_SOURCE_DIR"
    echo "   请确认阶段1数据收集已完成"
    exit 1
fi

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 复制文件
echo "正在复制数据文件..."
echo "  源目录: $DATA_SOURCE_DIR"
echo "  目标目录: $TARGET_DIR"
echo ""

# 复制 time_series.npy
if [ -f "$DATA_SOURCE_DIR/time_series.npy" ]; then
    cp "$DATA_SOURCE_DIR/time_series.npy" "$TARGET_DIR/time_series.npy"
    echo "✅ 已复制: time_series.npy"
else
    echo "❌ 警告：time_series.npy 不存在"
fi

# 复制 schedule_series.npy
if [ -f "$DATA_SOURCE_DIR/schedule_series.npy" ]; then
    cp "$DATA_SOURCE_DIR/schedule_series.npy" "$TARGET_DIR/schedule_series.npy"
    echo "✅ 已复制: schedule_series.npy"
else
    echo "❌ 警告：schedule_series.npy 不存在"
fi

# 复制 fault_history.pkl（阶段1新增）
if [ -f "$DATA_SOURCE_DIR/fault_history.pkl" ]; then
    cp "$DATA_SOURCE_DIR/fault_history.pkl" "$TARGET_DIR/fault_history.pkl"
    echo "✅ 已复制: fault_history.pkl（阶段1故障历史）"
else
    echo "⚠️  警告：fault_history.pkl 不存在（阶段1可能未生效）"
fi

echo ""
echo "✅ 数据复制完成！"
echo ""
echo "文件列表："
ls -lh "$TARGET_DIR" | grep -E "\.npy|\.pkl"
