# Directory paths
model_folder = 'recovery/CMODLBSrc/checkpoints/'
data_folder = 'recovery/PreGANSrc/data/'
data_filename = 'time_series.npy'

# Hyperparameters
num_epochs = 30
PERCENTILES = 90  # 保留作为备选方法

# 异常检测方法配置
ANOMALY_DETECTION_METHOD = 'multivariate'  # 可选: 'zscore', 'iqr', 'percentile', 'hybrid', 'multivariate', 'timeseries'
Z_SCORE_THRESHOLD = 1.8  # Z-score阈值，1.8对应约3.6%异常率（从2.0降低，增加异常样本，目标：提升精确度到50%）
IQR_MULTIPLIER = 1.5  # IQR倍数，标准箱线图方法
TIMESERIES_WINDOW_SIZE = 5  # 时间序列异常检测的窗口大小
HYBRID_VOTING = 'or'  # 混合方法的投票策略: 'and' (更严格，精确度高但可能无异常) 或 'or' (更宽松，召回率高)