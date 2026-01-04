#!/bin/bash
# 重复运行最佳配置实验（方案6 - MAMO-GAN）
# 配置: energy_weight=0.3, response_time_weight=0.3, migration_cost_weight=0.4
# migration_cost_threshold=130, cooldown_period=3, max_migrations_per_step=3
#
# 方法命名:
#   - FPE-GAN (Fault Prediction Encoder GAN): 原PreGAN方法
#   - TF-GAN (Transformer-based Fault GAN): 原PreGANPlus方法
#   - MAMO-GAN (Migration-Aware Multi-Objective GAN): 我们的改进方法（方案6配置）

# 激活conda环境
source ~/.zshrc 2>/dev/null || source ~/.bashrc 2>/dev/null
conda activate pregan

# 创建日志目录
LOG_DIR="experiment_logs"
mkdir -p $LOG_DIR

# 设置重复次数（默认3次）
REPEAT_COUNT=${1:-3}

echo "=========================================="
echo "重复运行最佳配置实验（方案6 - MAMO-GAN）"
echo "配置参数:"
echo "  - energy_weight = 0.3"
echo "  - response_time_weight = 0.3"
echo "  - migration_cost_weight = 0.4"
echo "  - migration_cost_threshold = 130"
echo "  - cooldown_period = 3"
echo "  - max_migrations_per_step = 3"
echo "重复次数: $REPEAT_COUNT"
echo "时间: $(date)"
echo "=========================================="
echo ""

# 检查配置是否正确
echo "检查配置..."
CONFIG_FILE="recovery/PreGANPlusEnhanced.py"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 错误: 找不到配置文件 $CONFIG_FILE"
    exit 1
fi

# 使用grep检查配置
if grep -q "energy_weight = 0.3" "$CONFIG_FILE" && \
   grep -q "response_time_weight = 0.3" "$CONFIG_FILE" && \
   grep -q "migration_cost_weight = 0.4" "$CONFIG_FILE" && \
   grep -q "migration_cost_threshold = 130" "$CONFIG_FILE" && \
   grep -q "cooldown_period = 3" "$CONFIG_FILE" && \
   grep -q "max_migrations_per_step = 3" "$CONFIG_FILE"; then
    echo "✅ 配置正确（方案6最佳配置）"
else
    echo "⚠️  警告: 配置可能不正确，请检查 $CONFIG_FILE"
    echo "预期配置:"
    echo "  - energy_weight = 0.3"
    echo "  - response_time_weight = 0.3"
    echo "  - migration_cost_weight = 0.4"
    echo "  - migration_cost_threshold = 130"
    echo "  - cooldown_period = 3"
    echo "  - max_migrations_per_step = 3"
    read -p "是否继续? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "操作已取消"
        exit 1
    fi
fi

echo ""

# 询问是否重置模型
echo "是否重置模型？(建议每次重复实验前重置模型以确保公平对比)"
read -p "重置模型? (y/n, 默认y): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "重置模型..."
    bash scripts/reset_models_for_weight_change.sh <<< "y"
    if [ $? -ne 0 ]; then
        echo "模型重置失败，请检查"
        exit 1
    fi
    echo ""
fi

# 运行重复实验
for i in $(seq 1 $REPEAT_COUNT); do
    echo "=========================================="
    echo "运行第 $i/$REPEAT_COUNT 次实验"
    echo "=========================================="
    
    # 生成时间戳
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/experiment_best_config_repeat${i}_${TIMESTAMP}.log"
    
    echo "日志文件: $LOG_FILE"
    echo ""
    
    # 运行实验
    python scripts/batch_run_experiments.py \
        --models PreGAN,PreGANPlus,PreGANPlusEnhanced \
        --steps 100 \
        2>&1 | tee $LOG_FILE
    
    # 检查退出状态
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo "✅ 第 $i 次实验完成"
        echo "日志文件: $LOG_FILE"
        echo ""
        
        # 生成对比图表
        echo "生成对比图表..."
        python grapher.py simulator
        echo ""
        
        # 如果不是最后一次，询问是否继续
        if [ $i -lt $REPEAT_COUNT ]; then
            echo "等待5秒后继续下一次实验..."
            sleep 5
            echo ""
        fi
    else
        echo ""
        echo "❌ 第 $i 次实验失败，请查看日志: $LOG_FILE"
        read -p "是否继续下一次实验? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "实验已中断"
            exit 1
        fi
    fi
done

echo ""
echo "=========================================="
echo "所有重复实验完成！"
echo "=========================================="
echo ""
echo "实验日志文件:"
ls -lt $LOG_DIR/experiment_best_config_repeat*.log 2>/dev/null | head -$REPEAT_COUNT
echo ""
echo "结果文件: all_datasets/simulator/"
echo "图表文件: results/simulator/"
echo ""
echo "建议: 对比多次实验的结果，检查结果的一致性"
echo "详细分析: 参见 docs/Experiments/Core/Repeat_Experiments_Analysis.md"
echo ""

