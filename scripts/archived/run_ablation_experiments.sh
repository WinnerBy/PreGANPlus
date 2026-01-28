#!/bin/bash
# 运行 MAMO-GAN 消融实验（可选训练 + 批量测试）

set -e

cd "$(dirname "$0")/.."

RUNS=1
TRAIN=0
METHODS=()
SAVE_DIR=""
LOG_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --train)
            TRAIN=1
            shift
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
            echo "  --runs N              每个方法运行次数（默认: 1）"
            echo "  --train               先进行阶段2+3训练（默认: 否）"
            echo "  --methods METHOD ...  指定方法列表"
            echo "  --save-dir DIR        数据保存目录"
            echo "  --log-dir DIR         日志保存目录"
            echo ""
            echo "可用方法:"
            echo "  PreGANPlusEnhanced"
            echo "  AblationNoTransformer AblationNoGAT"
            echo "  AblationNoMigrationAware AblationNoMultiObjective"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

ALL_METHODS=("PreGANPlusEnhanced" "AblationNoTransformer" "AblationNoGAT" "AblationNoMigrationAware" "AblationNoMultiObjective")
if [ ${#METHODS[@]} -eq 0 ]; then
    METHODS=("${ALL_METHODS[@]}")
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="experiment_data/ablation_${TIMESTAMP}"
fi
if [ -z "$LOG_DIR" ]; then
    LOG_DIR="experiment_logs/ablation_${TIMESTAMP}"
fi

mkdir -p "$SAVE_DIR" "$LOG_DIR"

echo "=================================================================================="
echo "MAMO-GAN 消融实验"
echo "=================================================================================="
echo "方法: $(IFS=' '; echo "${METHODS[*]}")"
echo "每个方法运行次数: $RUNS"
echo "是否训练: $TRAIN"
echo "数据保存目录: $SAVE_DIR"
echo "日志保存目录: $LOG_DIR"
echo ""

if [ "$TRAIN" -eq 1 ]; then
    echo "=================================================================================="
    echo "阶段2+3训练（可选）"
    echo "=================================================================================="
    for method in "${METHODS[@]}"; do
        echo ""
        echo "--- 训练 $method ---"
        python3 scripts/paper_experiment_stage3_gan_training.py --method "$method"
        python3 main.py -e "" -m 0 2>&1 | tee "$LOG_DIR/train_${method}_${TIMESTAMP}.log"
        echo "✅ $method 训练完成"
    done
    echo ""
fi

# 运行测试
TOTAL_TESTS=$((${#METHODS[@]} * RUNS))
CURRENT_TEST=0
for method in "${METHODS[@]}"; do
    mkdir -p "$SAVE_DIR/$method" "$LOG_DIR/$method"
    for run_id in $(seq 1 $RUNS); do
        CURRENT_TEST=$((CURRENT_TEST + 1))
        echo "=================================================================================="
        echo "[$CURRENT_TEST/$TOTAL_TESTS] 运行测试 $run_id: $method"
        echo "=================================================================================="

        python3 scripts/paper_experiment_stage4_testing.py --method "$method" 2>&1 | tee -a "$LOG_DIR/$method/configure.log"
        RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        LOG_FILE="$LOG_DIR/$method/run_$(printf "%03d" $run_id)_${RUN_TIMESTAMP}.log"

        if ! python3 main.py -e "" -m 0 > "$LOG_FILE" 2>&1; then
            echo "❌ 实验运行失败，检查日志: $LOG_FILE"
            continue
        fi

        echo "📦 保存实验数据..."
        LATEST_LOG_DIR=$(find logs -maxdepth 1 -name "RPiEdge_BWGD2_100_*" -type d | sort -r | head -1)
        if [ -z "$LATEST_LOG_DIR" ]; then
            echo "⚠️  未找到实验数据目录（RPiEdge_BWGD2_100_*）"
        else
            RUN_SAVE_DIR="$SAVE_DIR/$method/run_$(printf "%03d" $run_id)"
            mkdir -p "$RUN_SAVE_DIR"
            cp -r "$LATEST_LOG_DIR" "$RUN_SAVE_DIR/"
            echo "✅ 已保存到: $RUN_SAVE_DIR/$(basename "$LATEST_LOG_DIR")"
        fi
    done
done

echo ""
echo "=================================================================================="
echo "✅ 消融实验结束"
echo "=================================================================================="
echo "建议下一步："
echo "  python3 scripts/collect_ablation_results.py --data-dir \"$SAVE_DIR\""
