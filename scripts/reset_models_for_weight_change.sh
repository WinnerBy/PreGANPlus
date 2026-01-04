#!/bin/bash
# 脚本：重置模型以支持权重修改
# 用途：删除所有GAN相关的模型文件（Generator和Discriminator），以便在新权重下重新训练
# 注意：Encoder模型可以保留，因为它们不直接使用这些权重

echo "=========================================="
echo "重置模型以支持新模型架构"
echo "删除所有Generator和Discriminator模型"
echo "=========================================="
echo ""

# 确认操作
read -p "这将删除所有Generator和Discriminator模型。继续? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 1
fi

# PreGAN模型
echo "删除PreGAN模型..."
rm -f recovery/PreGANSrc/checkpoints/simulator_Gen_16.ckpt
rm -f recovery/PreGANSrc/checkpoints/simulator_Disc_16.ckpt
echo "✓ PreGAN Generator和Discriminator已删除"

# PreGANPlus模型
echo "删除PreGANPlus模型..."
rm -f recovery/PreGANSrc/checkpointsplus/simulator_Gen_16.ckpt
rm -f recovery/PreGANSrc/checkpointsplus/simulator_Disc_16.ckpt
echo "✓ PreGANPlus Generator和Discriminator已删除"

# PreGANPlusEnhanced模型（旧版本）
echo "删除PreGANPlusEnhanced旧模型..."
rm -f recovery/PreGANSrc/checkpointsplus/simulator_Gen_16_Attention.ckpt
rm -f recovery/PreGANSrc/checkpointsplus/simulator_Disc_16_MultiTask.ckpt
echo "✓ PreGANPlusEnhanced旧模型已删除"

# PreGANPlusEnhanced模型（新版本：迁移感知 + 多目标）
echo "删除PreGANPlusEnhanced新模型..."
rm -f recovery/PreGANSrc/checkpointsplus/simulator_Gen_16_MigrationAware.ckpt
rm -f recovery/PreGANSrc/checkpointsplus/simulator_Disc_16_MultiObjective.ckpt
echo "✓ PreGANPlusEnhanced新模型已删除"

echo ""
echo "=========================================="
echo "模型重置完成！"
echo "=========================================="
echo ""
echo "注意: Encoder模型已保留，因为它们不直接使用权重"
echo ""
echo "下一步: 运行实验重新训练所有模型"
echo "  方法1: 使用脚本"
echo "    bash run_experiment_new_weights.sh"
echo ""
echo "  方法2: 直接运行"
echo "    python scripts/batch_run_experiments.py --models PreGAN,PreGANPlus,PreGANPlusEnhanced --steps 100"
echo ""
echo "新模型架构:"
echo "  - Generator: Gen_16_MigrationAware (迁移感知)"
echo "  - Discriminator: Disc_16_MultiObjective (多目标)"
echo "  - 优化目标: 能量(0.4) + 响应时间(0.3) + 迁移成本(0.3)"

