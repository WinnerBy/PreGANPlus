import sys
sys.path.append('recovery/PreGANSrc/')

import os
import numpy as np
import torch
import torch.nn as nn

from .Recovery import *
from .PreGANSrc.src.constants import *
from .PreGANSrc.src.utils import *
from .PreGANSrc.src.train import *
from .PreGANSrc.src.train_multiobjective import train_gan_multiobjective
from .PreGANPlusEnhanced import PreGANPlusEnhancedRecovery


class _GenWithCostWrapper(nn.Module):
    """Wrap a standard generator to return (schedule, predicted_cost=0)."""
    def __init__(self, gen):
        super().__init__()
        self.gen = gen
        self.lr = getattr(gen, "lr", 0.00005)
        self.name = getattr(gen, "name", "GenWrapper")

    def forward(self, e, s):
        new_schedule = self.gen(e, s)
        predicted_cost = torch.tensor(0.0, dtype=new_schedule.dtype, device=new_schedule.device)
        return new_schedule, predicted_cost


class _GenScheduleOnlyWrapper(nn.Module):
    """Wrap a migration-aware generator to return schedule only."""
    def __init__(self, gen):
        super().__init__()
        self.gen = gen
        self.lr = getattr(gen, "lr", 0.00005)
        self.name = getattr(gen, "name", "GenScheduleOnly")

    def forward(self, e, s):
        new_schedule, _ = self.gen(e, s)
        return new_schedule


class AblationNoTransformerRecovery(PreGANPlusEnhancedRecovery):
    """
    Ablation: remove Transformer encoder, use FPE (GRU+GAT) encoder
    while keeping migration-aware generator + multi-objective discriminator.
    """
    def __init__(self, hosts, env, training=False):
        Recovery.__init__(self)
        self.model_name = f'FPE_{hosts}'
        self.gen_name = f'Gen_{hosts}_MigrationAware_ablation_notrans'
        self.disc_name = f'Disc_{hosts}_MultiObjective_ablation_notrans'
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.training = training
        self.save_gan = True

        # Multi-objective training hyperparameters
        self.energy_weight = 0.004
        self.response_time_weight = 0.14
        self.migration_cost_weight = 0.04
        self.sla_threshold = 2800.0
        self.migration_cost_threshold = 110

        # Migration control mechanisms
        self.migration_cooldown = {}
        self.cooldown_period = 10
        self.max_migrations_per_step = 2
        self.strict_migration_limit = 173
        self.total_migrations = 0

        self.load_models()

    def load_models(self):
        # Load encoder model (FPE)
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model_name)
        if self.epoch == -1:
            self.train_model()

        # Load enhanced GAN
        self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list = \
            self.load_gan_enhanced(model_plus_folder,
                                   f'{self.env_name}_{self.gen_name}.ckpt',
                                   f'{self.env_name}_{self.disc_name}.ckpt',
                                   self.gen_name, self.disc_name)
        self.gan_plotter = GAN_Plotter(self.env_name, self.gen_name, self.disc_name, self.training)
        self.ganloss = nn.BCELoss()
        self.train_time_data = load_npyfile(os.path.join(data_folder, self.env_name), data_filename)

    def train_model(self):
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
            save_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt',
                      self.model, self.optimizer, self.epoch, self.accuracy_list)


class AblationNoGATRecovery(PreGANPlusEnhancedRecovery):
    """
    Ablation: remove GAT, keep Transformer encoder and MAMO decision modules.
    """
    def __init__(self, hosts, env, training=False):
        Recovery.__init__(self)
        self.model_name = f'TransformerNoGAT_{hosts}'
        self.gen_name = f'Gen_{hosts}_MigrationAware_ablation_nogat'
        self.disc_name = f'Disc_{hosts}_MultiObjective_ablation_nogat'
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.training = training
        self.save_gan = True

        # Multi-objective training hyperparameters
        self.energy_weight = 0.004
        self.response_time_weight = 0.14
        self.migration_cost_weight = 0.04
        self.sla_threshold = 2800.0
        self.migration_cost_threshold = 110

        # Migration control mechanisms
        self.migration_cooldown = {}
        self.cooldown_period = 10
        self.max_migrations_per_step = 2
        self.strict_migration_limit = 173
        self.total_migrations = 0

        self.load_models()


class AblationNoMigrationAwareRecovery(PreGANPlusEnhancedRecovery):
    """
    Ablation: remove migration-aware generator, keep multi-objective discriminator.
    """
    def __init__(self, hosts, env, training=False):
        Recovery.__init__(self)
        self.model_name = f'Transformer_{hosts}'
        self.gen_name = f'Gen_{hosts}_ablation_nomigaware'
        self.disc_name = f'Disc_{hosts}_MultiObjective_ablation_nomigaware'
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.training = training
        self.save_gan = True

        # Multi-objective training hyperparameters
        self.energy_weight = 0.004
        self.response_time_weight = 0.14
        self.migration_cost_weight = 0.04
        self.sla_threshold = 2800.0
        self.migration_cost_threshold = 110

        # Migration control mechanisms (keep for consistency)
        self.migration_cooldown = {}
        self.cooldown_period = 10
        self.max_migrations_per_step = 2
        self.strict_migration_limit = 173
        self.total_migrations = 0

        self.load_models()

    def load_models(self):
        # Load encoder model
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_plus_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model_name)
        if self.epoch == -1:
            self.train_model()

        # Load standard generator + multi-objective discriminator
        gmodel, dmodel, gopt, dopt, epoch, accuracy_list = load_gan(
            model_plus_folder,
            f'{self.env_name}_{self.gen_name}.ckpt',
            f'{self.env_name}_{self.disc_name}.ckpt',
            f'Gen_{self.hosts}',
            f'Disc_{self.hosts}_MultiObjective'
        )
        self.gen = _GenWithCostWrapper(gmodel).double()
        self.disc = dmodel
        self.gopt, self.dopt = gopt, dopt
        self.epoch, self.accuracy_list = epoch, accuracy_list

        self.gan_plotter = GAN_Plotter(self.env_name, self.gen_name, self.disc_name, self.training)
        self.ganloss = nn.BCELoss()
        self.train_time_data = load_npyfile(os.path.join(data_folder, self.env_name), data_filename)


class AblationNoMultiObjectiveRecovery(PreGANPlusEnhancedRecovery):
    """
    Ablation: remove multi-objective discriminator, keep migration-aware generator.
    """
    def __init__(self, hosts, env, training=False):
        Recovery.__init__(self)
        self.model_name = f'Transformer_{hosts}'
        self.gen_name = f'Gen_{hosts}_MigrationAware_ablation_nomulti'
        self.disc_name = f'Disc_{hosts}_ablation_nomulti'
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.training = training
        self.save_gan = True

        # Training weights (not used in standard GAN training)
        self.energy_weight = 0.004
        self.response_time_weight = 0.14
        self.migration_cost_weight = 0.04
        self.sla_threshold = 2800.0
        self.migration_cost_threshold = 110

        # Migration control mechanisms
        self.migration_cooldown = {}
        self.cooldown_period = 10
        self.max_migrations_per_step = 2
        self.strict_migration_limit = 173
        self.total_migrations = 0

        self.load_models()

    def load_models(self):
        # Load encoder model
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_plus_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model_name)
        if self.epoch == -1:
            self.train_model()

        # Load migration-aware generator + standard discriminator
        gmodel, dmodel, gopt, dopt, epoch, accuracy_list = load_gan(
            model_plus_folder,
            f'{self.env_name}_{self.gen_name}.ckpt',
            f'{self.env_name}_{self.disc_name}.ckpt',
            f'Gen_{self.hosts}_MigrationAware',
            f'Disc_{self.hosts}'
        )
        self.gen = gmodel
        self.disc = dmodel
        self.gopt, self.dopt = gopt, dopt
        self.epoch, self.accuracy_list = epoch, accuracy_list

        self.gan_plotter = GAN_Plotter(self.env_name, self.gen_name, self.disc_name, self.training)
        self.ganloss = nn.BCELoss()
        self.train_time_data = load_npyfile(os.path.join(data_folder, self.env_name), data_filename)

    def train_gan(self, embedding, schedule_data):
        # Train discriminator (standard BCE)
        self.disc.zero_grad()
        new_schedule_data, _ = self.gen(embedding, schedule_data)
        probs = self.disc(schedule_data, new_schedule_data.detach())
        new_score, orig_score = run_simulation(self.env.stats, new_schedule_data), run_simulation(self.env.stats, schedule_data)
        true_probs = torch.tensor([0, 1], dtype=torch.double) if new_score <= orig_score else torch.tensor([1, 0], dtype=torch.double)
        disc_loss = self.ganloss(probs, true_probs.detach().clone())
        disc_loss.backward(); self.dopt.step()

        # Train generator
        self.gen.zero_grad()
        probs = self.disc(schedule_data, new_schedule_data)
        true_probs = torch.tensor([0, 1], dtype=torch.double)
        gen_loss = self.ganloss(probs, true_probs)
        gen_loss.backward(); self.gopt.step()

        # Append to accuracy list and save model
        if self.save_gan:
            self.epoch += 1
            self.accuracy_list.append((gen_loss.item(), disc_loss.item()))
            print(f'{color.HEADER}Epoch {self.epoch},\tGLoss = {gen_loss.item()},\tDLoss = {disc_loss.item()}{color.ENDC}')
            self.gan_plotter.plot(self.accuracy_list, self.epoch, new_score, orig_score)
            save_gan(model_plus_folder,
                    f'{self.env_name}_{self.gen_name}.ckpt',
                    f'{self.env_name}_{self.disc_name}.ckpt',
                    self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list)

    def recover_decision(self, embedding, schedule_data, original_decision):
        # Generator returns (new_schedule, predicted_migration_cost)
        new_schedule_data, predicted_migration_cost = self.gen(embedding, schedule_data)
        probs = self.disc(schedule_data, new_schedule_data)
        self.gan_plotter.new_better(probs[1] >= probs[0])
        if probs[0] > probs[1]:
            return original_decision

        # Form new decision with migration controls
        host_alloc = []
        container_alloc = [-1] * len(self.env.hostlist)
        for i in range(len(self.env.hostlist)):
            host_alloc.append([])
        for c in self.env.containerlist:
            if c and c.getHostID() != -1:
                host_alloc[c.getHostID()].append(c.id)
                container_alloc[c.id] = c.getHostID()

        potential_migrations = []
        decision_dict = dict(original_decision)
        current_epoch = self.epoch

        for cid in np.concatenate(host_alloc):
            cid = int(cid)
            one_hot = new_schedule_data[cid].tolist()
            new_host = one_hot.index(max(one_hot))
            orig_host = container_alloc[cid]

            if orig_host != new_host:
                priority = abs(new_schedule_data[cid][new_host] - schedule_data[cid][orig_host])
                if cid in self.migration_cooldown:
                    last_migration_epoch = self.migration_cooldown[cid]
                    if current_epoch - last_migration_epoch < self.cooldown_period:
                        continue
                potential_migrations.append((cid, new_host, priority))

        potential_migrations.sort(key=lambda x: x[2], reverse=True)
        allowed_migrations = potential_migrations[:self.max_migrations_per_step]

        if not self.training and hasattr(self, 'strict_migration_limit'):
            remaining_budget = self.strict_migration_limit - self.total_migrations
            if remaining_budget <= 0:
                allowed_migrations = []
            elif len(allowed_migrations) > remaining_budget:
                allowed_migrations = allowed_migrations[:remaining_budget]

        try:
            predicted_mc_value = float(predicted_migration_cost)
            if predicted_mc_value > self.migration_cost_threshold and len(allowed_migrations) > 1:
                allowed_migrations = allowed_migrations[:1]
        except Exception:
            pass

        hosts_from = [0] * self.hosts
        migration_count = 0
        for cid, new_host, priority in allowed_migrations:
            orig_host = container_alloc[cid]
            decision_dict[cid] = new_host
            hosts_from[orig_host] = 1
            migration_count += 1
            self.migration_cooldown[cid] = current_epoch

        if not self.training:
            self.total_migrations += migration_count

        self.gan_plotter.plot_test(hosts_from)
        return list(decision_dict.items())
