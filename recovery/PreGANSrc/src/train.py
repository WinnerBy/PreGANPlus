from .constants import *
from .utils import *
from .device_manager import get_device_manager
import torch.nn as nn
from tqdm import tqdm
from .plotter import *

anomaly_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss(reduction = 'mean')

num_zero, num_ones = 1, 1

# Model Training
def triplet_loss(anchor, positive_class, model):
	global PROTO_UPDATE_FACTOR
	# 使用 int 索引，兼容 list 与 numpy 标量；prototype 在 load_model 中已保证为 list
	idx = int(positive_class)
	if idx < 0 or idx >= len(model.prototype):
		return torch.tensor(0.0, dtype=anchor.dtype, device=anchor.device)
	positive_loss = mse_loss(anchor, model.prototype[idx].detach().clone())
	negative_class_list = [0, 1, 2]
	negative_class_list.remove(positive_class)
	negative_loss = []
	for nc in negative_class_list:
		nc_i = int(nc)
		if nc_i < len(model.prototype):
			negative_loss.append(mse_loss(anchor, model.prototype[nc_i]))
	if len(negative_loss) < 2:
		return torch.tensor(0.0, dtype=anchor.dtype, device=anchor.device)
	loss = positive_loss - torch.sum(torch.tensor(negative_loss, device=anchor.device))
	if positive_loss <= negative_loss[0] and positive_loss <= negative_loss[1]:
		factor = PROTO_UPDATE_FACTOR + PROTO_UPDATE_MIN
		model.prototype[idx] = factor * anchor + (1 - factor) * model.prototype[idx]
	return loss

def custom_loss(model, source, target_anomaly, target_class):
	global PROTO_UPDATE_FACTOR, num_ones, num_zero
	nz, no = 0, 0
	source_anomaly, source_prototype = source
	aloss, tloss = 0, torch.tensor(0, dtype=torch.float32)
	# 统一参数配置（用于所有编码器：PreGAN、PreGANPlus、Ablation）
	# 用服务器数据（4493样本，异常率69.4%）训练，150 epochs足够
	# 1. ANOMALY_WEIGHT=8：平衡点
	#    - PreGAN目标：75.74% → 77-79%（更好的异常率平衡）
	#    - PreGANPlus：保持99%+的高性能
	# 2. FOCAL_GAMMA=2.5：严格惩罚易分样本，提高判别精度
	# 3. 结果：参数统一，消融研究更清晰
	ANOMALY_WEIGHT = 8  # 统一参数：平衡精准率和召回率
	FOCAL_GAMMA = 2.5  # 统一参数：严格模式，关注边界样本
	
	for i, sa in enumerate(source_anomaly):
		# 计算基础损失
		base_loss = anomaly_loss(sa, torch.tensor([target_anomaly[i]], dtype=torch.long))
		
		# Focal Loss: 降低易分样本的权重，增加难分样本的权重
		# pt = 模型预测正确的概率
		pt = torch.exp(-base_loss)
		focal_weight = (1 - pt) ** FOCAL_GAMMA
		
		# 组合Focal Loss和类别权重
		multiplier = focal_weight.item() * (1 if target_anomaly[i] == 0 else ANOMALY_WEIGHT)
		nz += 1 if target_anomaly[i] == 0 else 0
		no += 1 if target_anomaly[i] == 1 else 0
		aloss += base_loss * multiplier
	for i, sp in enumerate(source_prototype):
		if target_anomaly[i] > 0:
			tloss += triplet_loss(sp, target_class[i], model)
	PROTO_UPDATE_FACTOR *= PROTO_FACTOR_DECAY
	num_zero += nz
	num_ones += no
	return aloss, tloss

def backprop(epoch, model, train_time_data, train_schedule_data, anomaly_data, class_data, optimizer, training = True):
	global PROTO_UPDATE_FACTOR, num_ones, num_zero
	num_zero, num_ones = 1, 1
	aloss_list, tloss_list = [], []
	for i in tqdm(range(train_time_data.shape[0]), leave=False, position=1):
		output = model(train_time_data[i], train_schedule_data[i])
		aloss, tloss = custom_loss(model, output, anomaly_data[i], class_data[i])
		aloss_list.append(aloss.item()); tloss_list.append(tloss.item())
		loss = aloss + tloss
		if training:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
	tqdm.write(f'Epoch {epoch},\tLoss = {np.mean(aloss_list)+np.mean(tloss_list)},\tALoss = {np.mean(aloss_list)},\tTLoss = {np.mean(tloss_list)}')
	factor = PROTO_UPDATE_FACTOR + PROTO_UPDATE_MIN
	return np.mean(aloss_list) + np.mean(tloss_list), factor

# Accuracy 
def anomaly_accuracy(source_anomaly, target_anomaly, model_plotter):
	correct = 0; res_list = []; tp, fp, tn, fn = 0, 0, 0, 0
	for i, sa in enumerate(source_anomaly):
		res = torch.argmax(sa).item() 
		res_list.append(res)
		if res == target_anomaly[i]:
			correct += 1
			if target_anomaly[i] == 1: tp += 1
			else: tn += 1
		else:
			if target_anomaly[i] == 1: fn += 1
			else: fp += 1
	if model_plotter is not None:
		model_plotter.update_anomaly(res_list, target_anomaly, correct/len(source_anomaly))
	return correct/len(source_anomaly), tp, tn, fp, fn

def _proto_at(model, idx):
	"""安全取 model.prototype[idx]，兼容 list 与 dict、numpy 标量。"""
	idx = int(idx)
	if isinstance(model.prototype, list):
		if 0 <= idx < len(model.prototype):
			return model.prototype[idx]
		return None
	return model.prototype.get(idx, model.prototype.get(str(idx), None))

def class_accuracy(source_prototype, target_anomaly, target_class, model, model_plotter):
	# 保证 prototype 为 list，避免 dict 导致 KeyError（与 load_model / triplet_loss 一致）
	if isinstance(model.prototype, dict):
		model.prototype = [model.prototype[k] for k in sorted(model.prototype.keys(), key=lambda x: int(x) if isinstance(x, str) else x)]
	correct, total = 0, 1e-4; prototypes = []
	for i, sp in enumerate(source_prototype):
		if target_anomaly[i] > 0:
			total += 1
			pidx = int(target_class[i])
			p_proto = _proto_at(model, pidx)
			if p_proto is None:
				continue
			positive_loss = mse_loss(sp, p_proto)
			negative_class_list = [0, 1, 2]
			negative_class_list.remove(pidx)
			negative_loss = []
			for nc in negative_class_list:
				nc_proto = _proto_at(model, nc)
				if nc_proto is not None:
					negative_loss.append(mse_loss(sp, nc_proto))
			if len(negative_loss) >= 2 and positive_loss <= negative_loss[0] and positive_loss <= negative_loss[1]:
				correct += 1
			prototypes.append((sp, pidx))
	if model_plotter is not None:
		model_plotter.update_class(prototypes, correct/total)
	return correct / total

def accuracy(model, train_time_data, train_schedule_data, anomaly_data, class_data, model_plotter):
	anomaly_correct, class_correct, class_total = 0, 0, 0; tpl, tnl, fpl, fnl = [], [], [], []
	for i, d in enumerate(train_time_data):
		output = model(train_time_data[i], train_schedule_data[i])
		source_anomaly, source_prototype = output
		res, tp, tn, fp, fn = anomaly_accuracy(source_anomaly, anomaly_data[i], model_plotter)
		anomaly_correct += res
		tpl.append(tp); tnl.append(tn); fpl.append(fp); fnl.append(fn)
		tp += res; fp += res; tn += (1 - res); fn += (1 - res)
		if np.sum(anomaly_data[i]) > 0:
			class_total += 1
			class_correct += class_accuracy(source_prototype, anomaly_data[i], class_data[i], model, model_plotter)
	tp, fp, tn, fn = np.mean(tpl), np.mean(fpl), np.mean(tnl), np.mean(fn)
	# 避免除零错误
	if tp + fp > 0:
		p = tp / (tp + fp)
	else:
		p = 0.0
	if tp + fn > 0:
		r = tp / (tp + fn)
	else:
		r = 0.0
	if p + r > 0:
		f1 = 2 * p * r / (p + r)
	else:
		f1 = 0.0
	tqdm.write(f'P = {p}, R = {r}, F1 = {f1}')
	# 避免除零错误
	if class_total > 0:
		class_score = class_correct / class_total
	else:
		class_score = 0.0
	return anomaly_correct / len(train_time_data), class_score

# Multi-task GAN Training Function
def train_gan_multitask(gen, disc, gopt, dopt, embedding, schedule_data, env, ganloss, 
                        score_regression_weight=0.5, score_generator_weight=0.3,
                        response_time_weight=0.2, sla_threshold=3000.0):
    """
    Multi-task GAN training function with response time constraint
    
    Args:
        gen: Generator model
        disc: Multi-task Discriminator model
        gopt: Generator optimizer
        dopt: Discriminator optimizer
        embedding: [16, 2] embedding from encoder
        schedule_data: [16, 16] current schedule
        env: Environment object (for run_simulation)
        ganloss: Loss function for classification (BCELoss)
        score_regression_weight: Weight for regression loss in discriminator training
        score_generator_weight: Weight for regression loss in generator training
        response_time_weight: Weight for response time constraint in generator training (新增)
        sla_threshold: SLA threshold for response time in seconds (新增)
    
    Returns:
        gen_loss: Generator loss
        disc_loss: Discriminator loss
        class_loss: Classification loss
        score_loss: Regression loss
        response_time_loss: Response time constraint loss (新增)
        new_score: Evaluation score of new schedule
        orig_score: Evaluation score of original schedule
        new_response_time: Response time of new schedule (新增)
        orig_response_time: Response time of original schedule (新增)
    """
    import torch
    from .utils import run_simulation
    
    # ========== Generate new schedule ==========
    new_schedule_data = gen(embedding, schedule_data)
    
    # ========== Real evaluation ==========
    new_score = run_simulation(env.stats, new_schedule_data)
    orig_score = run_simulation(env.stats, schedule_data)
    
    # ========== Get response times ==========
    # runSimulation returns (energy, latency) where latency is response time
    _, new_response_time = env.stats.runSimulation(new_schedule_data)
    _, orig_response_time = env.stats.runSimulation(schedule_data)
    
    # ========== Train Discriminator ==========
    disc.zero_grad()
    
    # Discriminator prediction (now returns 3 values)
    class_probs, score_pred, response_time_pred = disc(schedule_data, new_schedule_data.detach())
    
    # Task 1: Classification loss (judge which is better)
    true_class = torch.tensor([0, 1] if new_score <= orig_score else [1, 0], 
                              dtype=torch.float32, device=class_probs.device)
    class_loss = ganloss(class_probs, true_class)
    
    # Task 2: Regression loss (predict evaluation score)
    score_target = torch.tensor([new_score], dtype=torch.float32, device=score_pred.device)
    score_loss = mse_loss(score_pred, score_target)
    
    # Task 3: Response time prediction loss (新增)
    response_time_target = torch.tensor([new_response_time], dtype=torch.float32, device=response_time_pred.device)
    response_time_pred_loss = mse_loss(response_time_pred, response_time_target)
    
    # Multi-task loss (weighted combination)
    disc_loss = class_loss + score_regression_weight * score_loss + 0.1 * response_time_pred_loss
    
    disc_loss.backward()
    dopt.step()
    
    # ========== Train Generator ==========
    gen.zero_grad()
    
    # Discriminator prediction (not detached)
    class_probs_gen, score_pred_gen, response_time_pred_gen = disc(schedule_data, new_schedule_data)
    
    # Generator loss: encourage discriminator to think new schedule is better
    # Method 1: Classification loss
    target_better = torch.tensor([0, 1], dtype=torch.float32, device=class_probs_gen.device)
    gen_class_loss = ganloss(class_probs_gen, target_better)
    
    # Method 2: Regression loss (encourage predicting lower score)
    score_upper_bound = torch.tensor([orig_score], dtype=torch.float32, device=score_pred_gen.device)
    gen_score_loss = torch.relu(score_pred_gen - score_upper_bound + 0.1)
    
    # Method 3: Response time constraint loss (新增，关键优化)
    # Penalize if predicted response time exceeds SLA threshold
    sla_threshold_tensor = torch.tensor([sla_threshold], dtype=torch.float32, device=response_time_pred_gen.device)
    response_time_excess = torch.relu(response_time_pred_gen - sla_threshold_tensor)
    gen_response_time_loss = response_time_weight * response_time_excess
    
    # Also penalize if actual response time is high (additional constraint)
    actual_response_time_tensor = torch.tensor([new_response_time], dtype=torch.float32, device=response_time_pred_gen.device)
    actual_response_time_excess = torch.relu(actual_response_time_tensor - sla_threshold_tensor)
    gen_actual_response_time_loss = response_time_weight * 0.5 * actual_response_time_excess
    
    # Generator total loss
    gen_loss = gen_class_loss + score_generator_weight * gen_score_loss + gen_response_time_loss + gen_actual_response_time_loss
    
    gen_loss.backward()
    gopt.step()
    
    response_time_loss = gen_response_time_loss.item() + gen_actual_response_time_loss.item()
    
    return (gen_loss.item(), disc_loss.item(), class_loss.item(), score_loss.item(),
            response_time_loss, new_score, orig_score, new_response_time, orig_response_time)

