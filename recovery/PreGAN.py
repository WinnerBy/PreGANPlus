import sys
sys.path.append('recovery/PreGANSrc/')

import numpy as np
from copy import deepcopy
from .Recovery import *
from .PreGANSrc.src.constants import *
from .PreGANSrc.src.utils import *
from .PreGANSrc.src.train import *
from .PreGANSrc.src.device_manager import get_device_manager

class PreGANRecovery(Recovery):
    def __init__(self, hosts, env, training = False, encoder_only = False):
        super().__init__()
        self.model_name = f'FPE_{hosts}'
        self.gen_name = f'Gen_{hosts}'
        self.disc_name = f'Disc_{hosts}'
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.training = training
        self.encoder_only = encoder_only
        
        # FPE使用GRU，需要在CPU上运行（DGL不支持MPS）
        # 强制使用CPU设备
        self.device_manager = get_device_manager(verbose=True, force_cpu=True)
        self.device = self.device_manager.get_torch_device()
        
        self.load_models()

    def load_models(self):
        # Load encoder model
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model_name)
        # Train the model is not trained
        if self.epoch == -1: self.train_model()
        # Freeze encoder
        freeze(self.model)
        
        # 只在非encoder_only模式下加载GAN
        if not self.encoder_only:
            # Load generator and discriminator
            self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list = \
                load_gan(model_folder, f'{self.env_name}_{self.gen_name}.ckpt', f'{self.env_name}_{self.disc_name}.ckpt', self.gen_name, self.disc_name) 
            self.gan_plotter = GAN_Plotter(self.env_name, self.gen_name, self.disc_name, self.training)
            # Freeze GAN if not training
            if not self.training: freeze(self.gen); freeze(self.disc)
            if self.training:  self.ganloss = nn.BCELoss()
        
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
            save_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model, self.optimizer, self.epoch, self.accuracy_list)

    def train_gan(self, embedding, schedule_data):
        # encoder_only模式不训练GAN
        if self.encoder_only:
            return
            
        # 确保数据在正确的设备上
        embedding = embedding.to(self.device)
        schedule_data = schedule_data.to(self.device)
        
        # Train discriminator
        self.disc.zero_grad()
        new_schedule_data = self.gen(embedding, schedule_data)
        probs = self.disc(schedule_data, new_schedule_data.detach())
        new_score, orig_score = run_simulation(self.env.stats, new_schedule_data), run_simulation(self.env.stats, schedule_data)
        true_probs = torch.tensor([0, 1], dtype=torch.float32, device=self.device) if new_score <= orig_score else torch.tensor([1, 0], dtype=torch.float32, device=self.device)
        disc_loss = self.ganloss(probs, true_probs.detach().clone())
        disc_loss.backward(); self.dopt.step()
        # Train generator
        self.gen.zero_grad()
        probs = self.disc(schedule_data, new_schedule_data)
        true_probs = torch.tensor([0, 1], dtype=torch.float32, device=self.device) # to enforce new schedule is better than original schedule
        gen_loss = self.ganloss(probs, true_probs)
        gen_loss.backward(); self.gopt.step()
        # Append to accuracy list
        self.epoch += 1; self.accuracy_list.append((gen_loss.item(), disc_loss.item()))
        print(f'{color.HEADER}Epoch {self.epoch},\tGLoss = {gen_loss.item()},\tDLoss = {disc_loss.item()}{color.ENDC}')
        # Convert scores to scalars if they are tensors (for MPS compatibility)
        new_score_scalar = new_score.item() if hasattr(new_score, 'item') else new_score
        orig_score_scalar = orig_score.item() if hasattr(orig_score, 'item') else orig_score
        self.gan_plotter.plot(self.accuracy_list, self.epoch, new_score_scalar, orig_score_scalar)
        save_gan(model_folder, f'{self.env_name}_{self.gen_name}.ckpt', f'{self.env_name}_{self.disc_name}.ckpt', \
                self.gen, self.disc, self.gopt, self.dopt, self.epoch, self.accuracy_list)

    def recover_decision(self, embedding, schedule_data, original_decision):
        # 如果是encoder_only模式，直接返回原始决策
        if self.encoder_only:
            return original_decision
            
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

        
        # 将数据转换为torch tensor并移动到正确的设备
        time_data = torch.tensor(time_data, dtype=torch.float32, device=self.device)
        schedule_data = schedule_data.to(self.device)
    def run_encoder(self, schedule_data):
        # Get latest data from Stat
        time_data = self.env.stats.time_series
        time_data = normalize_test_time_data(time_data, self.train_time_data)
        if time_data.shape[0] >= self.model.n_window: time_data = time_data[-self.model.n_window:]
        time_data = convert_to_windows(time_data, self.model)[-1]
        anomaly, prototype = self.model(time_data, schedule_data)
        
        # DEBUG: Log encoder output
        import numpy as np
        import torch
        anomaly_sum = sum([torch.argmax(a).item() for a in anomaly])
        prototype_sum = sum([p.detach().cpu().numpy().sum() for p in prototype])
        print(f'[DEBUG PreGAN] Model: {self.model_name}, Anomaly sum: {anomaly_sum}, Prototype sum: {prototype_sum:.6f}')
        
        return anomaly, prototype

    def run_model(self, time_series, original_decision):
        # Run encoder
        dtype = self.device_manager.get_dtype()  # 获取兼容的dtype
        schedule_data = torch.tensor(self.env.scheduler.result_cache, dtype=dtype)
        anomaly, prototype = self.run_encoder(schedule_data)
        # Evaluate and print AScore/CScore on-the-fly (testing path)
        try:
            folder = os.path.join(data_folder, self.env_name)
            train_time_data, train_schedule_data, anomaly_data, class_data = \
                load_on_the_fly_dataset(self.model, folder, self.env.stats)
            anomaly_score, class_score = accuracy(self.model, train_time_data, train_schedule_data, anomaly_data, class_data, None)
            factor = PROTO_UPDATE_FACTOR + PROTO_UPDATE_MIN
            tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}')
        except Exception as e:
            # Non-fatal: continue if on-the-fly dataset or evaluation is unavailable
            pass
        # 异常检测：使用相对排序而不是绝对阈值
        # 编码器学到的是相对异常特征，找出异常概率最高的主机
        # 编码器输出的已经是Softmax概率，格式为 [[prob_normal, prob_anomaly]]
        
        anomaly_probs = []
        for i, a in enumerate(anomaly):
            try:
                # 编码器输出格式: [[prob_class0, prob_class1]]
                # 提取异常类（类1）的概率
                if len(a.shape) > 1:
                    anomaly_prob = a[0][1].item()  # [[normal, anomaly]] 格式
                else:
                    anomaly_prob = a[1].item() if len(a) > 1 else 0.0  # [normal, anomaly] 格式
            except Exception as e:
                print(f'[DEBUG PreGAN] Error parsing anomaly output for host {i}: {e}')
                anomaly_prob = 0.0
            anomaly_probs.append(anomaly_prob)
        
        # 调试：打印所有主机的异常分数
        print(f'[DEBUG PreGAN] All anomaly scores: {[f"{p:.3f}" for p in anomaly_probs]}')
        
        # 混合策略：绝对阈值 + 相对排序
        # 1. 首先过滤出异常概率超过绝对阈值的主机（编码器认为可能异常）
        ABSOLUTE_THRESHOLD = 0.3  # 异常概率至少要达到30%才考虑
        candidate_indices = [i for i, p in enumerate(anomaly_probs) if p > ABSOLUTE_THRESHOLD]
        
        if len(candidate_indices) > 0:
            # 2. 如果有超过阈值的主机，在这些候选中使用相对排序
            candidate_probs = [anomaly_probs[i] for i in candidate_indices]
            avg_candidate_prob = np.mean(candidate_probs)
            # 选择异常概率高于候选组平均值的主机
            anomaly_host_indices = [i for i in candidate_indices if anomaly_probs[i] > avg_candidate_prob]
            # 如果候选组内没有超过平均的（都差不多），选择概率最高的
            if len(anomaly_host_indices) == 0:
                anomaly_host_indices = [candidate_indices[np.argmax(candidate_probs)]]
            print(f'[DEBUG PreGAN] Candidates > {ABSOLUTE_THRESHOLD}: {len(candidate_indices)}, avg: {avg_candidate_prob:.3f}')
        else:
            # 3. 如果没有主机超过绝对阈值，说明系统整体正常，不触发GAN
            anomaly_host_indices = []
            print(f'[DEBUG PreGAN] No host exceeds absolute threshold {ABSOLUTE_THRESHOLD}')
        
        anomaly_detected = len(anomaly_host_indices) > 0
        
        if not anomaly_detected:
            print(f'[DEBUG PreGAN] No anomaly detected, returning original_decision')
            if not self.encoder_only:
                self.gan_plotter.update_anomaly_detected(0)
            return original_decision
        
        print(f'[DEBUG PreGAN] Anomaly detected in {len(anomaly_host_indices)} hosts (indices: {anomaly_host_indices}, scores: {[f"{anomaly_probs[i]:.3f}" for i in anomaly_host_indices]})')
        self.gan_plotter.update_anomaly_detected(1)
        
        # encoder_only模式：检测到异常后直接返回原始决策
        if self.encoder_only:
            return original_decision
            
        print(f'[DEBUG PreGAN] Proceeding with GAN')
        # Form prototype vectors for diagnosed hosts - 只为异常主机添加原型
        embedding = []
        for i, p in enumerate(prototype):
            if i in anomaly_host_indices:
                embedding.append(p)  # 异常主机使用实际原型
            else:
                embedding.append(torch.zeros_like(p))  # 正常主机使用零向量
        self.gan_plotter.update_class_detected(get_classes(embedding, self.model))
        embedding = torch.stack(embedding)
        # Pass through GAN
        if self.training:
            self.train_gan(embedding, schedule_data)
            # return original_decision
        return self.recover_decision(embedding, schedule_data, original_decision)

