# Directory paths
model_folder = 'recovery/PreGANSrc/checkpoints/'
model_plus_folder = 'recovery/PreGANSrc/checkpointsplus/'
data_folder = 'recovery/PreGANSrc/data/'
plot_folder = 'recovery/PreGANSrc/plots'
data_filename = 'time_series.npy'
schedule_filename = 'schedule_series.npy'

# Hyperparameters
num_epochs = 50
PERCENTILES = 90  # 保留作为备选方法

# 异常检测方法配置
# 分析发现：Percentile=90方法表现最好（27.7%精确度），进一步降低（85）反而降低精确度
# 为达到50%精确度目标，使用Percentile=90（最优阈值）+ 极高权重策略
ANOMALY_DETECTION_METHOD = 'percentile'  # 可选: 'zscore', 'iqr', 'percentile', 'hybrid', 'multivariate', 'timeseries'
PERCENTILES = 90  # 回到最优阈值（27.7%精确度），配合极高权重（100）来达到50%目标
Z_SCORE_THRESHOLD = 1.8  # Z-score阈值（用于其他方法）
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