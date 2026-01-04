"""
Multi-objective GAN training function
Balances energy, response time, and migration cost
"""
from .constants import *
from .utils import *
import torch.nn as nn
from .train import mse_loss

def calculate_migration_count(original_schedule, new_schedule):
    """
    计算从original_schedule到new_schedule的迁移次数
    
    Args:
        original_schedule: [16, 16] original schedule matrix
        new_schedule: [16, 16] new schedule matrix
    
    Returns:
        migration_count: int, number of migrations
    """
    migrations = 0
    for cid in range(original_schedule.shape[0]):
        orig_host = original_schedule[cid].argmax().item()
        new_host = new_schedule[cid].argmax().item()
        if orig_host != new_host:
            migrations += 1
    return migrations

def train_gan_multiobjective(gen, disc, gopt, dopt, embedding, schedule_data, env, ganloss,
                              energy_weight=0.3, response_time_weight=0.3, migration_cost_weight=0.4,
                              sla_threshold=2800.0, migration_cost_threshold=130):
    """
    多目标GAN训练函数，平衡能量、响应时间和迁移成本
    
    Args:
        gen: Generator model (Gen_16_MigrationAware)
        disc: Multi-objective Discriminator model (Disc_16_MultiObjective)
        gopt: Generator optimizer
        dopt: Discriminator optimizer
        embedding: [16, 2] embedding from encoder
        schedule_data: [16, 16] current schedule
        env: Environment object (for run_simulation)
        ganloss: Loss function for classification (BCELoss)
        energy_weight: Weight for energy optimization (default: 0.4)
        response_time_weight: Weight for response time constraint (default: 0.3)
        migration_cost_weight: Weight for migration cost constraint (default: 0.3)
        sla_threshold: SLA threshold for response time in seconds (default: 3000.0)
        migration_cost_threshold: Migration cost threshold (default: 180)
    
    Returns:
        gen_loss: Generator loss
        disc_loss: Discriminator loss
        class_loss: Classification loss
        energy_loss: Energy prediction loss
        response_time_loss: Response time prediction loss
        migration_cost_loss: Migration cost prediction loss
        gen_energy_loss: Generator energy constraint loss
        gen_response_time_loss: Generator response time constraint loss
        gen_migration_cost_loss: Generator migration cost constraint loss
        new_energy: Energy of new schedule
        orig_energy: Energy of original schedule
        new_response_time: Response time of new schedule
        orig_response_time: Response time of original schedule
        actual_migration_count: Actual migration count
        predicted_migration_cost: Predicted migration cost
    """
    import torch
    
    # ========== Generate new schedule ==========
    new_schedule_data, predicted_migration_cost = gen(embedding, schedule_data)
    
    # ========== Real evaluation ==========
    new_energy, new_response_time = env.stats.runSimulation(new_schedule_data)
    orig_energy, orig_response_time = env.stats.runSimulation(schedule_data)
    
    # ========== Calculate actual migration cost ==========
    actual_migration_count = calculate_migration_count(schedule_data, new_schedule_data)
    
    # ========== Train Discriminator ==========
    disc.zero_grad()
    
    # Discriminator prediction (returns 4 values)
    class_probs, energy_pred, response_time_pred, migration_cost_pred = \
        disc(schedule_data, new_schedule_data.detach())
    
    # Task 1: Classification loss (judge which is better)
    # 综合评分：能量 + 响应时间 + 迁移成本
    new_score = 0.8 * new_energy + 0.2 * new_response_time + 0.01 * actual_migration_count
    orig_score = 0.8 * orig_energy + 0.2 * orig_response_time + 0.01 * calculate_migration_count(schedule_data, schedule_data)
    true_class = torch.tensor([0, 1] if new_score <= orig_score else [1, 0], 
                              dtype=torch.double, device=class_probs.device)
    class_loss = ganloss(class_probs, true_class)
    
    # Task 2: Energy prediction loss
    energy_target = torch.tensor([new_energy], dtype=torch.double, device=energy_pred.device)
    energy_loss = mse_loss(energy_pred, energy_target)
    
    # Task 3: Response time prediction loss
    response_time_target = torch.tensor([new_response_time], dtype=torch.double, device=response_time_pred.device)
    response_time_pred_loss = mse_loss(response_time_pred, response_time_target)
    
    # Task 4: Migration cost prediction loss
    migration_cost_target = torch.tensor([actual_migration_count], dtype=torch.double, device=migration_cost_pred.device)
    migration_cost_pred_loss = mse_loss(migration_cost_pred, migration_cost_target)
    
    # Discriminator total loss (weighted combination)
    disc_loss = (class_loss + 
                 0.2 * energy_loss +
                 0.1 * response_time_pred_loss +
                 0.1 * migration_cost_pred_loss)
    
    disc_loss.backward()
    dopt.step()
    
    # ========== Train Generator ==========
    gen.zero_grad()
    
    # Discriminator prediction (not detached)
    class_probs_gen, energy_pred_gen, response_time_pred_gen, migration_cost_pred_gen = \
        disc(schedule_data, new_schedule_data)
    
    # Generator loss: balance multiple objectives
    # Method 1: Classification loss
    target_better = torch.tensor([0, 1], dtype=torch.double, device=class_probs_gen.device)
    gen_class_loss = ganloss(class_probs_gen, target_better)
    
    # Method 2: Energy constraint loss (encourage predicting lower energy)
    energy_upper_bound = torch.tensor([orig_energy], dtype=torch.double, device=energy_pred_gen.device)
    gen_energy_loss = torch.relu(energy_pred_gen - energy_upper_bound + 0.1)
    
    # Method 3: Response time constraint loss (penalize exceeding SLA threshold)
    sla_threshold_tensor = torch.tensor([sla_threshold], dtype=torch.double, device=response_time_pred_gen.device)
    response_time_excess = torch.relu(response_time_pred_gen - sla_threshold_tensor)
    gen_response_time_loss = response_time_weight * response_time_excess
    
    # Also penalize actual response time
    actual_response_time_tensor = torch.tensor([new_response_time], dtype=torch.double, device=response_time_pred_gen.device)
    actual_response_time_excess = torch.relu(actual_response_time_tensor - sla_threshold_tensor)
    gen_actual_response_time_loss = response_time_weight * 0.5 * actual_response_time_excess
    
    # Method 4: Migration cost constraint loss (进一步增强版，关键)
    # Penalize if predicted migration cost exceeds threshold
    migration_cost_threshold_tensor = torch.tensor([migration_cost_threshold], 
                                                    dtype=torch.double, device=migration_cost_pred_gen.device)
    migration_cost_excess = torch.relu(migration_cost_pred_gen - migration_cost_threshold_tensor)
    # 进一步增强迁移成本约束：使用立方惩罚，使超过阈值时惩罚更严重
    gen_migration_cost_loss = migration_cost_weight * (migration_cost_excess ** 3 + migration_cost_excess ** 2 + migration_cost_excess)
    
    # Also penalize actual migration cost (进一步增强版)
    actual_migration_cost_tensor = torch.tensor([actual_migration_count], 
                                                dtype=torch.double, device=migration_cost_pred_gen.device)
    actual_migration_cost_excess = torch.relu(actual_migration_cost_tensor - migration_cost_threshold_tensor)
    # Phase 1 optimization: 进一步增强实际迁移成本约束：使用立方惩罚，权重从1.5增加到2.0（更严格）
    gen_actual_migration_cost_loss = migration_cost_weight * 2.0 * (actual_migration_cost_excess ** 3 + actual_migration_cost_excess ** 2 + actual_migration_cost_excess)
    
    # Generator total loss: balance three objectives
    gen_loss = (gen_class_loss +
                energy_weight * gen_energy_loss +
                response_time_weight * (gen_response_time_loss + gen_actual_response_time_loss) +
                migration_cost_weight * (gen_migration_cost_loss + gen_actual_migration_cost_loss))
    
    gen_loss.backward()
    gopt.step()
    
    # Calculate total constraint losses for logging
    gen_response_time_total = gen_response_time_loss.item() + gen_actual_response_time_loss.item()
    gen_migration_cost_total = gen_migration_cost_loss.item() + gen_actual_migration_cost_loss.item()
    
    return (gen_loss.item(), disc_loss.item(), class_loss.item(),
            energy_loss.item(), response_time_pred_loss.item(), migration_cost_pred_loss.item(),
            gen_energy_loss.item(), gen_response_time_total, gen_migration_cost_total,
            new_energy, orig_energy, new_response_time, orig_response_time,
            actual_migration_count, predicted_migration_cost.item())

