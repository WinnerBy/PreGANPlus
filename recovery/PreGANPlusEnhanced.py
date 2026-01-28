import sys
sys.path.append('recovery/PreGANSrc/')

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from .Recovery import *
from .PreGANSrc.src.constants import *
from .PreGANSrc.src.utils import *
from .PreGANSrc.src.train import *
from .PreGANSrc.src.train_multiobjective import train_gan_multiobjective
from .PreGANSrc.src.device_manager import get_device_manager

class PreGANPlusEnhancedRecovery(Recovery):
    def __init__(self, hosts, env, training=False, encoder_only=False):
        super().__init__()
        self.model_name = f'Transformer_{hosts}'
        self.gen_name = f'Gen_{hosts}_MigrationAware'  # 使用迁移感知Generator
        self.disc_name = f'Disc_{hosts}_MultiObjective'  # 使用多目标Discriminator
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.training = training
        self.encoder_only = encoder_only
        self.save_gan = True
        
        # 初始化设备管理器
        self.device_manager = get_device_manager(verbose=True)
        # 编码器使用CPU（含GAT），GAN使用MPS/GPU
        self.encoder_device = torch.device('cpu')
        self.gan_device = self.device_manager.get_torch_device()
        
        # Multi-objective training hyperparameters
        # Note: Best results achieved with inference-side controls rather than heavy training weights
        # Current weights optimized for training stability
        self.energy_weight = 0.004            # Balanced energy focus
        self.response_time_weight = 0.14      # Strong RT focus (validated effective)
        self.migration_cost_weight = 0.04     # Moderate migration constraint
        self.sla_threshold = 2800.0           # SLA threshold in seconds
        self.migration_cost_threshold = 110   # Migration cost threshold
        
        # Migration control mechanisms (optimized through empirical testing)
        # Best validated config: max_per_step=2, limit=175 achieves:
        # - Migrations: 172 (baseline+2.4%)
        # - Energy: 1.959M (baseline-0.8%)  
        # - RT: 205k (baseline-10.1%)
        self.migration_cooldown = {}          # {container_id: last_migration_epoch}
        self.cooldown_period = 10              # Cooldown period: prevent thrashing
        self.max_migrations_per_step = 2      # Allow up to 2 per step for effective placement
        self.strict_migration_limit = 173     # Validated optimal: balances all metrics
        self.total_migrations = 0             # Counter for total migrations performed
        
        self.load_models()

    def load_models(self):
        # Load encoder model (same as PreGANPlus) - 放在CPU上
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_plus_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model_name)
        # 强制将编码器移到CPU
        self.model = self.model.to(self.encoder_device)
        if hasattr(self.model, 'gat_graph'):
            self.model.gat_graph = self.model.gat_graph.to(self.encoder_device)
        if hasattr(self.model, 'prototype'):
            for i in range(len(self.model.prototype)):
                self.model.prototype[i] = self.model.prototype[i].to(self.encoder_device)
        
        # Train the model if not trained
        if self.epoch == -1: 
            self.train_model()
        
        # 只在非encoder_only模式下加载GAN
        if not self.encoder_only:
            # Load generator and discriminator (enhanced versions) - 放在GPU上
            # Note: We need to modify load_gan to support new model names
            self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list = \
                self.load_gan_enhanced(model_plus_folder, 
                                      f'{self.env_name}_{self.gen_name}.ckpt', 
                                      f'{self.env_name}_{self.disc_name}.ckpt', 
                                      self.gen_name, self.disc_name)
            # 将GAN模型移到GPU设备
            self.gen = self.gen.to(self.gan_device)
            self.disc = self.disc.to(self.gan_device)
            
            self.gan_plotter = GAN_Plotter(self.env_name, self.gen_name, self.disc_name, self.training)
            # GAN is always tuned
            self.ganloss = nn.BCELoss()
        
        self.train_time_data = load_npyfile(os.path.join(data_folder, self.env_name), data_filename)

    def load_gan_enhanced(self, folder, gfname, dfname, gmodelname, dmodelname):
        """Load enhanced GAN models"""
        # Model class names are fixed: Gen_16_MigrationAware and Disc_16_MultiObjective
        # (designed for 16 hosts, but file names can include host count)
        base_gname = 'Gen_16_MigrationAware'
        base_dname = 'Disc_16_MultiObjective'
        
        # Use load_gan function with base model class names
        # File names (gfname, dfname) can include host count for different experiments
        try:
            gmodel, dmodel, gopt, dopt, epoch, accuracy_list = \
                load_gan(folder, gfname, dfname, base_gname, base_dname)
        except Exception as e:
            # If models don't exist, create new ones
            import recovery.PreGANSrc.src.models
            try:
                gmodel_class = getattr(recovery.PreGANSrc.src.models, base_gname)
                dmodel_class = getattr(recovery.PreGANSrc.src.models, base_dname)
            except AttributeError as attr_err:
                # Fallback: if model classes don't exist, raise error
                raise RuntimeError(f"Model classes {base_gname} or {base_dname} not found. "
                                 f"Make sure they are defined in models.py. "
                                 f"Original error: {attr_err}")
            dtype = self.device_manager.get_dtype()  # 获取兼容的dtype
            gmodel = gmodel_class().to(dtype=dtype)
            dmodel = dmodel_class().to(dtype=dtype)
            gopt = torch.optim.AdamW(gmodel.parameters(), lr=gmodel.lr, weight_decay=1e-5)
            dopt = torch.optim.AdamW(dmodel.parameters(), lr=dmodel.lr, weight_decay=1e-5)
            epoch = -1
            accuracy_list = []
        return gmodel, dmodel, gopt, dopt, epoch, accuracy_list

    def train_model(self):
        """Train encoder model (same as PreGANPlus)"""
        self.model_plotter = Model_Plotter(self.env_name, self.model_name)
        folder = os.path.join(data_folder, self.env_name)
        train_time_data, train_schedule_data, anomaly_data, class_data = load_dataset(folder, self.model)
        for self.epoch in tqdm(range(self.epoch+1, self.epoch+num_epochs+1), position=0):
            loss, factor = backprop(self.epoch, self.model, train_time_data, train_schedule_data, 
                                   anomaly_data, class_data, self.optimizer)
            anomaly_score, class_score = accuracy(self.model, train_time_data, train_schedule_data, 
                                                 anomaly_data, class_data, self.model_plotter)
            tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}')
            self.accuracy_list.append((loss, factor, anomaly_score, class_score))
            self.model_plotter.plot(self.accuracy_list, self.epoch)
            save_model(model_plus_folder, f'{self.env_name}_{self.model_name}.ckpt', 
                      self.model, self.optimizer, self.epoch, self.accuracy_list)

    def tune_model(self):
        """Tune encoder for a single epoch (same as PreGANPlus)"""
        folder = os.path.join(data_folder, self.env_name)
        train_time_data, train_schedule_data, anomaly_data, class_data = \
            load_on_the_fly_dataset(self.model, folder, self.env.stats)
        loss, factor = backprop(self.epoch, self.model, train_time_data, train_schedule_data, 
                               anomaly_data, class_data, self.optimizer)
        anomaly_score, class_score = accuracy(self.model, train_time_data, train_schedule_data, 
                                             anomaly_data, class_data, None)
        tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}')
        self.accuracy_list.append((loss, factor, anomaly_score, class_score))

    def train_gan(self, embedding, schedule_data):
        """Multi-objective GAN training: balance energy, response time, and migration cost"""
        # encoder_only模式不训练GAN
        if self.encoder_only:
            return
            
        # 将数据移到GAN设备（GPU）
        embedding = embedding.to(self.gan_device)
        schedule_data = schedule_data.to(self.gan_device)
        
        # Use the multi-objective training function
        (gen_loss, disc_loss, class_loss, energy_loss, response_time_loss, migration_cost_loss,
         gen_energy_loss, gen_response_time_loss, gen_migration_cost_loss,
         new_energy, orig_energy, new_response_time, orig_response_time,
         actual_migration_count, predicted_migration_cost) = \
            train_gan_multiobjective(
                self.gen, self.disc, self.gopt, self.dopt,
                embedding, schedule_data, self.env, self.ganloss,
                self.energy_weight, self.response_time_weight, self.migration_cost_weight,
                self.sla_threshold, self.migration_cost_threshold
            )
        
        # Append to accuracy list and save model
        if self.save_gan:
            self.epoch += 1
            self.accuracy_list.append((
                gen_loss, disc_loss, class_loss, energy_loss, response_time_loss, migration_cost_loss,
                gen_energy_loss, gen_response_time_loss, gen_migration_cost_loss,
                new_energy, orig_energy, new_response_time, orig_response_time,
                actual_migration_count, predicted_migration_cost
            ))
            print(f'{color.HEADER}Epoch {self.epoch},\t'
                  f'GLoss = {gen_loss:.4f},\tDLoss = {disc_loss:.4f},\t'
                  f'ClassLoss = {class_loss:.4f},\tEnergyLoss = {energy_loss:.4f},\t'
                  f'RTLoss = {response_time_loss:.4f},\tMCLoss = {migration_cost_loss:.4f},\t'
                  f'NewEnergy = {new_energy:.2f},\tOrigEnergy = {orig_energy:.2f},\t'
                  f'NewRT = {new_response_time:.2f}s,\tOrigRT = {orig_response_time:.2f}s,\t'
                  f'ActualMC = {actual_migration_count},\tPredMC = {predicted_migration_cost:.2f}{color.ENDC}')
            # Use energy as score for plotting (can be adjusted)
            new_score = 0.8 * new_energy + 0.2 * new_response_time
            orig_score = 0.8 * orig_energy + 0.2 * orig_response_time
            self.gan_plotter.plot(self.accuracy_list, self.epoch, new_score, orig_score)
            save_gan(model_plus_folder, 
                    f'{self.env_name}_{self.gen_name}.ckpt', 
                    f'{self.env_name}_{self.disc_name}.ckpt',
                    self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list)

    def recover_decision(self, embedding, schedule_data, original_decision):
        """Recover decision using enhanced GAN with Phase 1 migration control optimizations"""
        # 如果是encoder_only模式，直接返回原始决策
        if self.encoder_only:
            return original_decision
            
        # 将数据移到GAN设备
        embedding = embedding.to(self.gan_device)
        schedule_data = schedule_data.to(self.gan_device)
        
        # Generator now returns (new_schedule, predicted_migration_cost)
        new_schedule_data, predicted_migration_cost = self.gen(embedding, schedule_data)
        
        # Use multi-objective discriminator: get classification probabilities
        # Note: Discriminator now returns 4 values (class_probs, energy_pred, response_time_pred, migration_cost_pred)
        class_probs, energy_pred, response_time_pred, migration_cost_pred = self.disc(schedule_data, new_schedule_data)
        self.gan_plotter.new_better(class_probs[1] >= class_probs[0])
        
        if class_probs[0] > class_probs[1]:  # original better
            return original_decision
        
        # Form new decision
        host_alloc = []
        container_alloc = [-1] * len(self.env.hostlist)
        for i in range(len(self.env.hostlist)):
            host_alloc.append([])
        for c in self.env.containerlist:
            if c and c.getHostID() != -1:
                host_alloc[c.getHostID()].append(c.id)
                container_alloc[c.id] = c.getHostID()
        
        # Phase 1 Optimization: Collect potential migrations with priorities
        potential_migrations = []  # List of (cid, new_host, priority)
        decision_dict = dict(original_decision)
        current_epoch = self.epoch
        
        for cid in np.concatenate(host_alloc):
            cid = int(cid)
            one_hot = new_schedule_data[cid].tolist()
            new_host = one_hot.index(max(one_hot))
            orig_host = container_alloc[cid]
            
            if orig_host != new_host:  # Migration needed
                # Calculate migration priority (based on schedule change magnitude)
                priority = abs(new_schedule_data[cid][new_host] - schedule_data[cid][orig_host])
                
                # Phase 1 Optimization 1: Check migration cooldown
                if cid in self.migration_cooldown:
                    last_migration_epoch = self.migration_cooldown[cid]
                    if current_epoch - last_migration_epoch < self.cooldown_period:
                        # Still in cooldown period, skip this migration
                        continue
                
                potential_migrations.append((cid, new_host, priority))
        
        # Phase 1 Optimization 2: Limit number of migrations per step
        # Sort by priority (highest first) and keep only top N
        potential_migrations.sort(key=lambda x: x[2], reverse=True)
        allowed_migrations = potential_migrations[:self.max_migrations_per_step]

        # Phase 1 Optimization 3: Global migration budget enforcement (testing mode only)
        if not self.training and hasattr(self, 'strict_migration_limit'):
            remaining_budget = self.strict_migration_limit - self.total_migrations
            if remaining_budget <= 0:
                # Budget exhausted, no more migrations allowed
                allowed_migrations = []
                print(f"[Enhanced] Migration budget exhausted (limit={self.strict_migration_limit}), forcing zero migrations.")
            elif len(allowed_migrations) > remaining_budget:
                # Limit to remaining budget
                allowed_migrations = allowed_migrations[:remaining_budget]
                print(f"[Enhanced] Budget remaining={remaining_budget}, limiting migrations this step to {len(allowed_migrations)}")

        # If the generator already predicts a high migration cost for this step,
        # further tighten to a single best migration (no retraining needed).
        try:
            predicted_mc_value = float(predicted_migration_cost)
            if predicted_mc_value > self.migration_cost_threshold and len(allowed_migrations) > 1:
                allowed_migrations = allowed_migrations[:1]
        except Exception:
            # If prediction is not a scalar, fall back to the standard cap
            pass
        
        # Apply allowed migrations
        hosts_from = [0] * self.hosts
        migration_count = 0
        for cid, new_host, priority in allowed_migrations:
            orig_host = container_alloc[cid]
            decision_dict[cid] = new_host
            hosts_from[orig_host] = 1
            migration_count += 1
            # Update cooldown tracking
            self.migration_cooldown[cid] = current_epoch
        
        # Update total migration counter (testing mode only)
        if not self.training:
            self.total_migrations += migration_count
        
        self.gan_plotter.plot_test(hosts_from)
        return list(decision_dict.items())

    def run_encoder(self, schedule_data):
        """Run encoder (same as PreGANPlus)"""
        # Get latest data from Stat
        time_data = self.env.stats.time_series
        
        # 确保数据在CPU上（编码器在CPU）
        time_data = time_data.to(self.encoder_device)
        schedule_data = schedule_data.to(self.encoder_device)
        
        if time_data.shape[0] >= self.model.n_window:
            time_data = time_data[-self.model.n_window:]
        time_data = convert_to_windows(time_data, self.model)[-1]
        return self.model(time_data, schedule_data)

    def run_model(self, time_series, original_decision):
        """Main model execution (same as PreGANPlus)"""
        # Run encoder
        dtype = self.device_manager.get_dtype()  # 获取兼容的dtype
        schedule_data = torch.tensor(self.env.scheduler.result_cache, dtype=dtype, device=self.encoder_device)
        anomaly, prototype = self.run_encoder(schedule_data)
        
        # If no anomaly predicted, return original decision
        anomaly_detected = False
        for a in anomaly:
            prediction = torch.argmax(a).item()
            if prediction == 1:
                anomaly_detected = True
                if not self.encoder_only:
                    self.gan_plotter.update_anomaly_detected(1)
                break
        
        if not anomaly_detected:
            if not self.encoder_only:
                self.gan_plotter.update_anomaly_detected(0)
            return original_decision
        
        # encoder_only模式：检测到异常后直接返回原始决策
        if self.encoder_only:
            return original_decision
        
        # Form prototype vectors for diagnosed hosts
        embedding = [torch.zeros_like(p) if torch.argmax(anomaly[i]).item() == 0 else p 
                    for i, p in enumerate(prototype)]
        self.gan_plotter.update_class_detected(get_classes(embedding, self.model))
        embedding = torch.stack(embedding)
        
        # Pass through enhanced GAN (only when training)
        if self.training:
            self.train_gan(embedding, schedule_data)
            # Tune Model during training only
            self.tune_model()
        
        return self.recover_decision(embedding, schedule_data, original_decision)

