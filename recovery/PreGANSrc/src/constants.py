# Directory paths
model_folder = 'recovery/PreGANSrc/checkpoints/'
model_plus_folder = 'recovery/PreGANSrc/checkpointsplus/'
data_folder = 'recovery/PreGANSrc/data/'
plot_folder = 'recovery/PreGANSrc/plots'
data_filename = 'time_series.npy'
schedule_filename = 'schedule_series.npy'

# Hyperparameters
num_epochs = 100  # 从50增加到100，给模型更多学习时间
PERCENTILES = 98  # 从95提高到98，极度严格（仅标记最上层2%）

# 异常检测方法配置
# 改进：使用multivariate方法而非单维percentile
# 原因：真实故障通常涉及多个维度（CPU+RAM）同时升高，multivariate更好地捕捉这一特征
ANOMALY_DETECTION_METHOD = 'multivariate'  # 改为multivariate，更好处理多维异常
Z_SCORE_THRESHOLD = 3.0  # 从2.0提高到3.0，更严格（仅标记3倍标准差之外）

# 数据增强配置（针对类别不平衡）
AUGMENT_ANOMALY_SAMPLES = True  # 启用异常样本增强
AUGMENT_FACTOR = 3  # 将异常样本重复3次
IQR_MULTIPLIER = 1.5  # IQR倍数，标准箱线图方法
TIMESERIES_WINDOW_SIZE = 5  # 时间序列异常检测的窗口大小
HYBRID_VOTING = 'or'  # 混合方法的投票策略: 'and' (更严格，精确度高但可能无异常) 或 'or' (更宽松，召回率高)
PROTO_DIM = 2
PROTO_UPDATE_FACTOR = 0.2
PROTO_UPDATE_MIN = 0.02
PROTO_FACTOR_DECAY = 0.995
LATEST_WINDOW_SIZE = 10

# GAN parameters
# Back to original 0.8/0.2 for maximum energy efficiency
# We will optimize model architecture to improve SLA instead of changing weights
Coeff_Energy = 0.8
Coeff_Latency = 0.2