import os
import torch
import numpy as np
from .constants import *
from .models import *

def convert_to_windows(data, model):
	data = torch.tensor(data).double()
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w)
	return torch.stack(windows)

def form_test_dataset(data, method=None, **kwargs):
	"""
	异常检测数据集生成函数
	支持多种异常检测方法：
	- 'zscore': 基于Z-score的统计异常检测
	- 'iqr': 基于IQR的鲁棒异常检测
	- 'percentile': 基于百分位数的传统方法
	- 'multivariate': 多维组合异常检测（推荐用于提升精确度）
	- 'timeseries': 时间序列异常检测
	- 'hybrid': 混合方法（多维+时间序列，最推荐）
	"""
	from .constants import ANOMALY_DETECTION_METHOD, Z_SCORE_THRESHOLD, IQR_MULTIPLIER, PERCENTILES, TIMESERIES_WINDOW_SIZE, HYBRID_VOTING
	
	# 如果没有指定方法，使用配置中的方法
	if method is None:
		method = ANOMALY_DETECTION_METHOD
	
	# 根据方法选择异常检测策略
	if method == 'zscore':
		z_threshold = kwargs.get('z_threshold', Z_SCORE_THRESHOLD)
		anomaly_per_dim = _detect_anomaly_zscore(data, z_threshold)
	elif method == 'iqr':
		iqr_multiplier = kwargs.get('iqr_multiplier', IQR_MULTIPLIER)
		anomaly_per_dim = _detect_anomaly_iqr(data, iqr_multiplier)
	elif method == 'percentile':
		percentile = kwargs.get('percentile', PERCENTILES)
		anomaly_per_dim = data > np.percentile(data, percentile, axis=0)
	elif method == 'multivariate':
		z_threshold = kwargs.get('z_threshold', Z_SCORE_THRESHOLD)
		anomaly_per_dim = _detect_anomaly_multivariate(data, z_threshold)
	elif method == 'timeseries':
		z_threshold = kwargs.get('z_threshold', Z_SCORE_THRESHOLD)
		window_size = kwargs.get('window_size', TIMESERIES_WINDOW_SIZE)
		anomaly_per_dim = _detect_anomaly_timeseries(data, window_size, z_threshold)
	elif method == 'hybrid':
		z_threshold = kwargs.get('z_threshold', Z_SCORE_THRESHOLD)
		window_size = kwargs.get('window_size', TIMESERIES_WINDOW_SIZE)
		voting = kwargs.get('voting', HYBRID_VOTING)
		anomaly_per_dim = _detect_anomaly_hybrid(data, z_threshold, window_size, voting)
	else:
		raise ValueError(f"Unknown anomaly detection method: {method}")
	
	# 生成异常标签
	anomaly_which_dim, anomaly_any_dim = [], []
	for i in range(0, data.shape[1], 3):
		anomaly_which_dim.append(np.argmax(data[:, i:i+3] + 0, axis=1))
		anomaly_any_dim.append(np.logical_or.reduce(anomaly_per_dim[:, i:i+3], axis=1))
	anomaly_any_dim = np.stack(anomaly_any_dim, axis=1)
	anomaly_which_dim = np.stack(anomaly_which_dim, axis=1)
	
	# 调试信息：统计异常样本数量
	total_samples = anomaly_any_dim.size
	anomaly_samples = np.sum(anomaly_any_dim)
	anomaly_rate = anomaly_samples / total_samples if total_samples > 0 else 0.0
	print(f"[异常检测] 方法: {method}, 总样本数: {total_samples}, 异常样本数: {anomaly_samples}, 异常率: {anomaly_rate:.2%}")
	
	# 如果异常率太低，给出警告
	if anomaly_rate < 0.01:  # 小于1%
		print(f"[警告] 异常率过低 ({anomaly_rate:.2%})，可能导致训练困难。建议：")
		print(f"  - 降低Z_SCORE_THRESHOLD（当前: {kwargs.get('z_threshold', 'default')}）")
		print(f"  - 或使用更宽松的方法（如'zscore'或'multivariate'）")
		print(f"  - 或使用'hybrid'方法时设置HYBRID_VOTING='or'")
	
	return anomaly_any_dim + 0, anomaly_which_dim

def _detect_anomaly_zscore(data, z_threshold=2.5):
	"""
	基于Z-score的异常检测
	原理: |value - mean| > z_threshold * std
	z_threshold=2.5 对应约1.2%异常率（正态分布假设）
	"""
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0) + 1e-8  # 避免除零
	z_scores = np.abs((data - mean) / std)
	return z_scores > z_threshold

def _detect_anomaly_iqr(data, iqr_multiplier=1.5):
	"""
	基于IQR的鲁棒异常检测
	原理: value < Q1 - k*IQR or value > Q3 + k*IQR
	对异常值更鲁棒，适合偏态分布
	"""
	anomaly_per_dim = np.zeros_like(data, dtype=bool)
	
	for dim in range(data.shape[1]):
		q1 = np.percentile(data[:, dim], 25)
		q3 = np.percentile(data[:, dim], 75)
		iqr = q3 - q1
		
		if iqr > 0:  # 避免IQR为0的情况
			lower_bound = q1 - iqr_multiplier * iqr
			upper_bound = q3 + iqr_multiplier * iqr
			anomaly_per_dim[:, dim] = (data[:, dim] < lower_bound) | (data[:, dim] > upper_bound)
		# 如果IQR为0，该维度不标记异常
	
	return anomaly_per_dim

def _detect_anomaly_multivariate(data, z_threshold=2.0):
	"""
	多维组合异常检测（改进版）
	考虑CPU、内存、带宽的组合异常，使用Mahalanobis距离
	改进：
	1. 使用更严格的阈值（z_threshold * 1.3）
	2. 添加时间平滑（连续异常才标记）
	3. 更准确地识别真正的故障
	"""
	from scipy.spatial.distance import mahalanobis
	from scipy.linalg import inv
	
	n_timesteps, n_features = data.shape
	n_hosts = n_features // 3
	anomaly_per_dim = np.zeros_like(data, dtype=bool)
	
	# 使用更严格的阈值（提升精确度）
	z_threshold_strict = z_threshold * 1.3  # 2.0 * 1.3 = 2.6
	# 时间平滑：需要连续异常的时间步数
	min_consecutive_anomalies = 2
	
	for host_idx in range(n_hosts):
		start_idx = host_idx * 3
		end_idx = start_idx + 3
		host_data = data[:, start_idx:end_idx]  # [timesteps, 3] (CPU, Memory, Bandwidth)
		
		# 计算协方差矩阵
		cov = np.cov(host_data.T)
		mean = np.mean(host_data, axis=0)
		
		try:
			cov_inv = inv(cov)
		except:
			# 如果矩阵不可逆，使用伪逆
			cov_inv = np.linalg.pinv(cov)
		
		# 计算Mahalanobis距离
		distances = []
		for t in range(n_timesteps):
			try:
				dist = mahalanobis(host_data[t], mean, cov_inv)
			except:
				# 如果计算失败，使用欧氏距离
				dist = np.linalg.norm(host_data[t] - mean)
			distances.append(dist)
		
		distances = np.array(distances)
		
		# 使用Z-score方法判断异常（使用更严格的阈值）
		dist_mean = np.mean(distances)
		dist_std = np.std(distances) + 1e-8
		z_scores = np.abs((distances - dist_mean) / dist_std)
		
		# 初步标记异常时间步（使用更严格的阈值）
		anomaly_timesteps_raw = z_scores > z_threshold_strict
		
		# 时间平滑：只有连续异常才标记为异常（减少瞬时噪声）
		anomaly_timesteps = np.zeros_like(anomaly_timesteps_raw, dtype=bool)
		consecutive_count = 0
		for t in range(n_timesteps):
			if anomaly_timesteps_raw[t]:
				consecutive_count += 1
				if consecutive_count >= min_consecutive_anomalies:
					# 标记当前及之前的连续异常时间步
					for i in range(max(0, t - consecutive_count + 1), t + 1):
						anomaly_timesteps[i] = True
			else:
				consecutive_count = 0
		
		# 如果时间步异常，标记该主机的所有特征为异常
		for t in range(n_timesteps):
			if anomaly_timesteps[t]:
				anomaly_per_dim[t, start_idx:end_idx] = True
	
	return anomaly_per_dim

def _detect_anomaly_timeseries(data, window_size=5, z_threshold=2.0):
	"""
	时间序列异常检测
	使用滑动窗口计算局部统计量，考虑时间上下文
	"""
	n_timesteps, n_features = data.shape
	anomaly_per_dim = np.zeros_like(data, dtype=bool)
	
	for dim in range(n_features):
		for t in range(n_timesteps):
			# 滑动窗口
			start = max(0, t - window_size // 2)
			end = min(n_timesteps, t + window_size // 2 + 1)
			window_data = data[start:end, dim]
			
			# 计算窗口内的统计量
			mean = np.mean(window_data)
			std = np.std(window_data) + 1e-8
			
			# Z-score
			z_score = abs((data[t, dim] - mean) / std)
			anomaly_per_dim[t, dim] = z_score > z_threshold
	
	return anomaly_per_dim

def _detect_anomaly_hybrid(data, z_threshold=2.0, window_size=5, voting='and'):
	"""
	混合异常检测方法
	结合多维组合检测和时间序列检测
	voting: 'and' (两种方法都认为异常，更严格) 或 'or' (任一方法认为异常，更宽松)
	"""
	# 多维检测
	anomaly_multivariate = _detect_anomaly_multivariate(data, z_threshold)
	
	# 时间序列检测
	anomaly_timeseries = _detect_anomaly_timeseries(data, window_size, z_threshold)
	
	# 投票
	if voting == 'and':
		# 两种方法都认为异常才标记为异常（更严格，精确度更高）
		anomaly_per_dim = anomaly_multivariate & anomaly_timeseries
	else:
		# 任一方法认为异常就标记为异常（更宽松，召回率更高）
		anomaly_per_dim = anomaly_multivariate | anomaly_timeseries
	
	return anomaly_per_dim

def load_npyfile(folder, fname):
	path = os.path.join(folder, fname)
	if not os.path.exists(path):
		raise Exception('Data not found ' + path)
	return np.load(path)

def load_dataset(folder, model):
	time_data = load_npyfile(folder, data_filename)
	time_data = normalize_time_data(time_data) # Normalize data
	train_time_data = convert_to_windows(time_data, model)
	return train_time_data

def save_model(folder, fname, model, optimizer, epoch, accuracy_list):
	path = os.path.join(folder, fname)
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy_list': accuracy_list}, path)

def load_model(folder, fname, modelname):
	import recovery.CMODLBSrc.src.models
	path = os.path.join(folder, fname)
	model_class = getattr(recovery.CMODLBSrc.src.models, modelname)
	model = model_class().double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	if os.path.exists(path):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, epoch, accuracy_list

# Misc
def normalize_time_data(time_data):
	# 使用Z-score标准化替代最大值归一化，更好地处理异常值
	mean = np.mean(time_data, axis=0)
	std = np.std(time_data, axis=0) + 1e-8  # 添加小值避免除零
	return (time_data - mean) / std 

def normalize_test_time_data(time_data, train_time_data):
	# 使用训练数据的均值和标准差进行Z-score标准化
	mean = np.mean(train_time_data, axis=0)
	std = np.std(train_time_data, axis=0) + 1e-8
	return (time_data - mean) / std

def freeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = False

def unfreeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = True

class color:
	HEADER = '\033[95m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	RED = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'