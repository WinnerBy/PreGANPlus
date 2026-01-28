import sys
sys.path.append('recovery/PreGANSrc/')

import numpy as np
from copy import deepcopy
from .Recovery import *
from .PreGANSrc.src.constants import *
from .PreGANSrc.src.utils import *
from .PreGANSrc.src.train import *
from .PreGANSrc.src.device_manager import get_device_manager

class PreGANPlusRecovery(Recovery):
    def __init__(self, hosts, env, training = False, encoder_only = False):
        super().__init__()
        self.model_name = f'Transformer_{hosts}'
        self.gen_name = f'Gen_{hosts}'
        self.disc_name = f'Disc_{hosts}'
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
        
        self.load_models()

    def load_models(self):
        # Load encoder model - 放在CPU上（含GAT）
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_plus_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model_name)
        # 强制将编码器移到CPU
        self.model = self.model.to(self.encoder_device)
        if hasattr(self.model, 'gat_graph'):
            self.model.gat_graph = self.model.gat_graph.to(self.encoder_device)
        if hasattr(self.model, 'prototype'):
            for i in range(len(self.model.prototype)):
                self.model.prototype[i] = self.model.prototype[i].to(self.encoder_device)
        
        # Train the model if not trained (offline training same as PreGAN)
        if self.epoch == -1: self.train_model()
        
        # 只在非encoder_only模式下加载GAN
        if not self.encoder_only:
            # Load generator and discriminator - 放在GPU上
            self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list = \
                load_gan(model_plus_folder, f'{self.env_name}_{self.gen_name}.ckpt', f'{self.env_name}_{self.disc_name}.ckpt', self.gen_name, self.disc_name)
            # 将GAN模型移到GPU设备
            self.gen = self.gen.to(self.gan_device)
            self.disc = self.disc.to(self.gan_device)
            
            self.gan_plotter = GAN_Plotter(self.env_name, self.gen_name, self.disc_name, self.training)
            # GAN is always tuned
            self.ganloss = nn.BCELoss()
        
        self.train_time_data = load_npyfile(os.path.join(data_folder, self.env_name), data_filename)

    def train_model(self):
        self.model_plotter = Model_Plotter(self.env_name, self.model_name)
        folder = os.path.join(data_folder, self.env_name)
        train_time_data, train_schedule_data, anomaly_data, class_data = load_dataset(folder, self.model)
        for self.epoch in tqdm(range(self.epoch+1, self.epoch+num_epochs+1), position=0):
            loss, factor = backprop(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.optimizer)
            anomaly_score, class_score = accuracy(self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.model_plotter)
            tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}')
            self.accuracy_list.append((loss, factor, anomaly_score, class_score))
            self.model_plotter.plot(self.accuracy_list, self.epoch)
            save_model(model_plus_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model, self.optimizer, self.epoch, self.accuracy_list)

    def tune_model(self):
        # tune for a single epoch
        folder = os.path.join(data_folder, self.env_name)
        train_time_data, train_schedule_data, anomaly_data, class_data = load_on_the_fly_dataset(self.model, folder, self.env.stats)
        loss, factor = backprop(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.optimizer)
        anomaly_score, class_score = accuracy(self.model, train_time_data, train_schedule_data, anomaly_data, class_data, None)
        tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}')
        self.accuracy_list.append((loss, factor, anomaly_score, class_score))

    def train_gan(self, embedding, schedule_data):
        # encoder_only模式不训练GAN
        if self.encoder_only:
            return
            
        # 将数据移到GAN设备（GPU）
        embedding = embedding.to(self.gan_device)
        schedule_data = schedule_data.to(self.gan_device)
        
        # Train discriminator
        self.disc.zero_grad()
        new_schedule_data = self.gen(embedding, schedule_data)
        probs = self.disc(schedule_data, new_schedule_data.detach())
        new_score, orig_score = run_simulation(self.env.stats, new_schedule_data), run_simulation(self.env.stats, schedule_data)
        true_probs = torch.tensor([0, 1], dtype=torch.double, device=self.gan_device) if new_score <= orig_score else torch.tensor([1, 0], dtype=torch.double, device=self.gan_device)
        disc_loss = self.ganloss(probs, true_probs.detach().clone())
        disc_loss.backward(); self.dopt.step()
        # Train generator
        self.gen.zero_grad()
        probs = self.disc(schedule_data, new_schedule_data)
        true_probs = torch.tensor([0, 1], dtype=torch.double, device=self.gan_device) # to enforce new schedule is better than original schedule
        gen_loss = self.ganloss(probs, true_probs)
        gen_loss.backward(); self.gopt.step()
        # Append to accuracy list and save model
        if self.save_gan:            
            self.epoch += 1; self.accuracy_list.append((gen_loss.item(), disc_loss.item()))
            print(f'{color.HEADER}Epoch {self.epoch},\tGLoss = {gen_loss.item()},\tDLoss = {disc_loss.item()}{color.ENDC}')
            self.gan_plotter.plot(self.accuracy_list, self.epoch, new_score, orig_score)
            save_gan(model_plus_folder, f'{self.env_name}_{self.gen_name}.ckpt', f'{self.env_name}_{self.disc_name}.ckpt', \
                    self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list)

    def recover_decision(self, embedding, schedule_data, original_decision):
        # 如果是encoder_only模式，直接返回原始决策
        if self.encoder_only:
            return original_decision
            
        # 将数据移到GAN设备
        embedding = embedding.to(self.gan_device)
        schedule_data = schedule_data.to(self.gan_device)
        
        new_schedule_data = self.gen(embedding, schedule_data)
        probs = self.disc(schedule_data, new_schedule_data)
        self.gan_plotter.new_better(probs[1] >= probs[0])
        if probs[0] > probs[1]: # original better
            return original_decision
        # Form new decision
        host_alloc = []; container_alloc = [-1] * len(self.env.hostlist)
        for i in range(len(self.env.hostlist)): host_alloc.append([])
        for c in self.env.containerlist:
            if c and c.getHostID() != -1: 
                host_alloc[c.getHostID()].append(c.id) 
                container_alloc[c.id] = c.getHostID()
        decision_dict = dict(original_decision); hosts_from = [0] * self.hosts
        for cid in np.concatenate(host_alloc):
            cid = int(cid)
            one_hot = schedule_data[cid].tolist()
            new_host = one_hot.index(max(one_hot))
            if container_alloc[cid] != new_host: 
                decision_dict[cid] = new_host
                hosts_from[container_alloc[cid]] = 1
        self.gan_plotter.plot_test(hosts_from)
        return list(decision_dict.items())

    def run_encoder(self, schedule_data):
        # Get latest data from Stat
        time_data = self.env.stats.time_series
        time_data = normalize_test_time_data(time_data, self.train_time_data)
        if time_data.shape[0] >= self.model.n_window: time_data = time_data[-self.model.n_window:]
        time_data = convert_to_windows(time_data, self.model)[-1]
        
        # 确保数据在CPU上（编码器在CPU）
        time_data = time_data.to(self.encoder_device)
        schedule_data = schedule_data.to(self.encoder_device)
        
        anomaly, prototype = self.model(time_data, schedule_data)
        
        # DEBUG: Log encoder output
        import numpy as np
        import torch
        anomaly_sum = sum([torch.argmax(a).item() for a in anomaly])
        prototype_sum = sum([p.detach().cpu().numpy().sum() for p in prototype])
        print(f'[DEBUG PreGANPlus] Model: {self.model_name}, Anomaly sum: {anomaly_sum}, Prototype sum: {prototype_sum:.6f}')
        
        return anomaly, prototype

    def run_model(self, time_series, original_decision):
        # Run encoder (在CPU上)
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
            print(f'[DEBUG PreGANPlus] No anomaly detected, returning original_decision')
            if not self.encoder_only:
                self.gan_plotter.update_anomaly_detected(0)
            return original_decision
        
        # encoder_only模式：检测到异常后直接返回原始决策
        if self.encoder_only:
            return original_decision
            
        print(f'[DEBUG PreGANPlus] Anomaly detected, proceeding with GAN')
        # Form prototype vectors for diagnosed hosts
        embedding = [torch.zeros_like(p) if torch.argmax(anomaly[i]).item() == 0 else p for i, p in enumerate(prototype)]
        self.gan_plotter.update_class_detected(get_classes(embedding, self.model))
        embedding = torch.stack(embedding)
        # Pass through GAN (only when training)
        if self.training:
            self.train_gan(embedding, schedule_data)
            # Tune Model during training only
            self.tune_model()
        return self.recover_decision(embedding, schedule_data, original_decision)

