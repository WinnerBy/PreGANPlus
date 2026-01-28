#!/bin/bash
# 多次运行stage4测试脚本
# 可以指定运行哪些方法，也可以运行全部方法
# 每次运行后自动保存数据和日志

set -e

# 进入项目根目录
cd "$(dirname "$0")/.."

# 默认配置
RUNS=3
METHODS=()
SAVE_DIR=""
LOG_DIR=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --methods)
            shift
            METHODS=()
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                METHODS+=("$1")
                shift
            done
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --runs N              每个方法运行的次数（默认: 3）"
            echo "  --methods METHOD ...  要测试的方法列表（默认: 所有方法）"
            echo "  --save-dir DIR        数据保存目录（默认: experiment_data/stage4_TIMESTAMP）"
            echo "  --log-dir DIR         日志保存目录（默认: experiment_logs/stage4_TIMESTAMP）"
            echo "  --help, -h            显示此帮助信息"
            echo ""
            echo "可用方法: PreGAN PreGANPlus PreGANPlusEnhanced CMODLB DFTM ECLB PCFT"
            echo ""
            echo "示例:"
            echo "  # 运行所有方法，每个3次"
            echo "  $0 --runs 3"
            echo ""
            echo "  # 只运行PreGAN和PreGANPlus，每个5次"
            echo "  $0 --methods PreGAN PreGANPlus --runs 5"
            echo ""
            echo "  # 只运行PreGANPlusEnhanced，10次"
            echo "  $0 --methods PreGANPlusEnhanced --runs 10"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 所有可用方法
ALL_METHODS=("PreGAN" "PreGANPlus" "PreGANPlusEnhanced" "CMODLB" "DFTM" "ECLB" "PCFT")

# 确定要运行的方法
if [ ${#METHODS[@]} -eq 0 ]; then
    METHODS=("${ALL_METHODS[@]}")
fi

# 验证方法列表
if [ ${#METHODS[@]} -eq 0 ]; then
    echo "错误: 没有指定要运行的方法"
    echo "使用 --help 查看帮助信息"
    exit 1
fi

# 创建保存目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="experiment_data/stage4_${TIMESTAMP}"
fi
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="experiment_logs/stage4_${TIMESTAMP}"
fi

mkdir -p "$SAVE_DIR"
mkdir -p "$LOG_DIR"

# 保存配置
METHODS_JSON=$(printf '"%s",' "${METHODS[@]}" | sed 's/,$//')
cat > "$SAVE_DIR/config.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "methods": [$METHODS_JSON],
  "runs_per_method": $RUNS,
  "save_dir": "$SAVE_DIR",
  "log_dir": "$LOG_DIR"
}
EOF

echo "=================================================================================="
echo "多次运行 Stage4 测试"
echo "=================================================================================="
echo "方法: $(IFS=' '; echo "${METHODS[*]}")"
echo "每个方法运行次数: $RUNS"
echo "数据保存目录: $SAVE_DIR"
echo "日志保存目录: $LOG_DIR"
echo ""

# 计算总测试数
TOTAL_TESTS=$((${#METHODS[@]} * RUNS))
CURRENT_TEST=0

# 运行测试
for method in "${METHODS[@]}"; do
    # 为每个方法创建目录
    mkdir -p "$SAVE_DIR/$method"
    mkdir -p "$LOG_DIR/$method"
    
    for run_id in $(seq 1 $RUNS); do
        CURRENT_TEST=$((CURRENT_TEST + 1))
        
        echo "=================================================================================="
        echo "[$CURRENT_TEST/$TOTAL_TESTS] 运行测试 $run_id: $method"
        echo "=================================================================================="
        
        # 配置stage4（直接修改main.py）
        echo "📝 配置 $method..."
        if ! python3 scripts/paper_experiment_stage4_testing.py --method "$method" 2>&1 | tee -a "$LOG_DIR/$method/configure.log"; then
            echo "❌ 配置失败"
            continue
        fi
        
        # 运行main.py
        echo "🚀 运行实验..."
        RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        LOG_FILE="$LOG_DIR/$method/run_$(printf "%03d" $run_id)_${RUN_TIMESTAMP}.log"
        
        if ! python3 main.py -e "" -m 0 > "$LOG_FILE" 2>&1; then
            echo "❌ 实验运行失败，检查日志: $LOG_FILE"
            continue
        fi
        
        # 保存实验数据
        echo "📦 保存实验数据..."
        LATEST_LOG_DIR=$(find logs -maxdepth 1 -name "RPiEdge_BWGD2_100_*" -type d | sort -r | head -1)
        
        if [ -z "$LATEST_LOG_DIR" ]; then
            echo "⚠️  未找到实验数据目录（RPiEdge_BWGD2_100_*）"
        else
            RUN_SAVE_DIR="$SAVE_DIR/$method/run_$(printf "%03d" $run_id)"
            mkdir -p "$RUN_SAVE_DIR"
            
            if cp -r "$LATEST_LOG_DIR" "$RUN_SAVE_DIR/"; then
                echo "✅ 实验数据已保存到: $RUN_SAVE_DIR/$(basename "$LATEST_LOG_DIR")"
            else
                echo "❌ 保存数据时出错"
            fi
        fi
        
        echo "✅ $method 测试完成 (运行 $run_id/$RUNS)"
        echo ""
    done
done

# 生成汇总
cat > "$SAVE_DIR/summary.txt" <<EOF
==================================================================================
Stage4 多次运行汇总
==================================================================================
运行时间: $(date)
方法: ${METHODS[*]}
每个方法运行次数: $RUNS
总运行次数: $TOTAL_TESTS

数据保存目录: $SAVE_DIR
日志保存目录: $LOG_DIR

各方法运行统计:
EOF

for method in "${METHODS[@]}"; do
    RUN_COUNT=$(find "$SAVE_DIR/$method" -maxdepth 1 -type d -name "run_*" | wc -l)
    echo "  $method: $RUN_COUNT/$RUNS 次成功" >> "$SAVE_DIR/summary.txt"
done

cat >> "$SAVE_DIR/summary.txt" <<EOF

==================================================================================
下一步:
1. 检查保存的数据和日志
2. 使用 analyze_stage4_results.py 分析结果:
   python3 scripts/analyze_stage4_results.py --data-dir $SAVE_DIR
3. 根据预期标准挑选最佳结果
4. 使用grapher.py或其他工具进行分析和绘图
==================================================================================
EOF

echo "=================================================================================="
echo "📊 所有测试结果汇总"
echo "=================================================================================="
cat "$SAVE_DIR/summary.txt"
echo ""
echo "✅ 所有测试完成！"
echo ""
