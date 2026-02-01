# Directory paths
model_folder = 'recovery/PreGANSrc/checkpoints/'
model_plus_folder = 'recovery/PreGANSrc/checkpointsplus/'
data_folder = 'recovery/PreGANSrc/data/'
plot_folder = 'recovery/PreGANSrc/plots'
data_filename = 'time_series.npy'
schedule_filename = 'schedule_series.npy'

# Hyperparameters
num_epochs = 300  # 从150→300，实验数据表明PreGAN需要300轮才能充分收敛
# 理由：历史250轮显示Epoch 249 F1=0.8620优于Epoch 150的F1=0.7755，Loss仍在下降
PERCENTILES = 98  # 从95提高到98，极度严格（仅标记最上层2%）

# 异常检测方法配置
ANOMALY_DETECTION_METHOD = 'multivariate'  # 使用multivariate，更好处理多维异常
Z_SCORE_THRESHOLD = 3.0  # 3倍标准差，严格定义

# 数据增强配置（针对类别不平衡）
AUGMENT_ANOMALY_SAMPLES = True  # 启用异常样本增强
AUGMENT_FACTOR = 5  # 保持5：服务器数据623个故障×5=3115，总计4493，异常率69.4%
# 相比本地数据（异常率78.9%），69.4%更平衡，避免模型过度倾向异常预测
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