import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from .constants import *
from .dlutils import *

## FPE
class FPE_16(nn.Module):
    def __init__(self):
        super(FPE_16, self).__init__()
        self.name = 'FPE_16'
        self.lr = 0.0001
        self.n_hosts = 16  # 图节点数量：16个主机
        self.n_feats = 3 * self.n_hosts  # 48 (3 metrics × 16 hosts)
        self.n_window = 3  # 时间窗口大小
        self.n_latent = 10  # 每个主机的潜在维度
        self.n_hidden = 16
        
        # GRU配置：输入48维特征
        self.gru = nn.GRU(
            input_size=self.n_feats,
            hidden_size=self.n_window,
            num_layers=1, 
            batch_first=False
        )
        
        # GAT配置：图节点数=16
        src_ids = torch.tensor([i for i in range(self.n_hosts) for _ in range(self.n_hosts)])
        dst_ids = torch.tensor([j for _ in range(self.n_hosts) for j in range(self.n_hosts)])
        self.gat_graph = dgl.graph((src_ids, dst_ids))  # 16节点全连接图
        
        # GAT输入特征维度=3，输出特征维度=16
        self.gat_input_feats = 3
        self.gat_output_feats = self.n_hosts
        self.gat = GAT(
            self.gat_graph,
            self.gat_input_feats,
            self.gat_output_feats
        )
        
        # 多头注意力：输入维度=3+16=19
        self.mha = nn.MultiheadAttention(
            embed_dim=self.n_window + self.gat_output_feats,
            num_heads=1
        )
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(self.n_window * (self.n_window + self.gat_output_feats), self.n_hosts * self.n_latent),
            nn.LeakyReLU(True),
        )
        
        self.anomaly_decoder = nn.Sequential(
            nn.Linear(self.n_latent, 2), 
            nn.Softmax(dim=0),
        )
        self.prototype_decoder = nn.Sequential(
            nn.Linear(self.n_latent, PROTO_DIM), 
            nn.Sigmoid(),
        )
        self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

    def encode(self, t, s):
        # 输入t形状：[3, 48]（3个时间步，16主机×3特征）
        t_batch = t.unsqueeze(1)  # [3, 1, 48]（添加batch维度）
        
        # GRU处理：输出[3, 1, 3]（3时间步，1 batch，3特征）
        h0 = torch.randn(1, 1, self.n_window, dtype=torch.double, device=t.device)
        gru_out, _ = self.gru(t_batch, h0)
        
        # GAT处理
        # 1. 重塑输入特征为[3, 1, 16, 3]（时间步，batch，节点，特征）
        gat_input = t_batch.view(self.n_window, 1, self.n_hosts, self.gat_input_feats)
        
        # 2. 转置为[batch=1, seq_len=3, nodes=16, feats=3]
        gat_input = torch.transpose(gat_input, 0, 1)  # [1, 3, 16, 3]
        
        # 3. GAT前向传播（输出[1, 3, 16, 16]）
        gat_out = self.gat(gat_input)
        
        # 4. 关键修复：调整GAT输出维度与GRU输出匹配（从4维变为3维）
        # 移除节点维度的冗余信息，通过均值聚合节点特征
        gat_out = gat_out.mean(dim=2)  # [1, 3, 16]（聚合16个节点的特征）
        gat_out = torch.transpose(gat_out, 0, 1)  # [3, 1, 16]（与GRU输出维度一致）
        
        # 拼接GRU和GAT输出：两者均为3维 [3, 1, 3] + [3, 1, 16] = [3, 1, 19]
        concat_out = torch.cat((gru_out, gat_out), dim=2)
        
        # 多头注意力
        attn_out, _ = self.mha(concat_out, concat_out, concat_out)
        
        # 编码到latent空间
        attn_flat = attn_out.squeeze(1).view(-1)  # 3×19=57
        latent = self.encoder(attn_flat).view(self.n_hosts, self.n_latent)
        return latent

    def anomaly_decode(self, t):
        anomaly_scores = []
        for elem in t:
            anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))    
        return anomaly_scores

    def prototype_decode(self, t):
        prototypes = []
        for elem in t:
            prototypes.append(self.prototype_decoder(elem))    
        return prototypes

    def forward(self, t, s):
        latent = self.encode(t, s)
        anomaly_scores = self.anomaly_decode(latent)
        prototypes = self.prototype_decode(latent)
        return anomaly_scores, prototypes

# Generator Network (保持不变，因为FPE输出维度未变)
class Gen_16(nn.Module):
    def __init__(self):
        super(Gen_16, self).__init__()
        self.name = 'Gen_16'
        self.lr = 0.00005
        self.n_hosts = 16
        self.n_hidden = 64
        self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
        self.delta = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
        )

    def forward(self, e, s):
        del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
        return s + del_s.reshape(self.n_hosts, self.n_hosts)

# Discriminator Network (保持不变)
class Disc_16(nn.Module):
    def __init__(self):
        super(Disc_16, self).__init__()
        self.name = 'Disc_16'
        self.lr = 0.00005
        self.n_hosts = 16
        self.n_hidden = 64
        self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
        self.probs = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
        )

    def forward(self, o, n):
        probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
        return probs


## 以下为其他模型（FPE_50, Attention_50, Gen_50, Disc_50, Transformer_16）保持不变
## FPE
class FPE_50(nn.Module):
    def __init__(self):
        super(FPE_50, self).__init__()
        self.name = 'FPE_50'
        self.lr = 0.0001
        self.n_hosts = 50
        self.n_feats = 3 * self.n_hosts
        self.n_window = 3 # w_size = 5
        self.n_latent = 10
        self.n_hidden = 50
        self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
        self.gru = nn.GRU(self.n_window, self.n_window, 1)
        src_ids = torch.tensor(list(range(self.n_feats))); dst_ids = torch.tensor([self.n_feats] * self.n_feats)
        self.gat = GAT(dgl.graph((src_ids, dst_ids)), self.n_window, self.n_window)
        self.mha = nn.MultiheadAttention(self.n_feats * 2 + 1, 1)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_window * (self.n_feats * 2 + 1), self.n_hosts * self.n_latent), nn.LeakyReLU(True),
        )
        self.anomaly_decoder = nn.Sequential(
            nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
        )
        self.prototype_decoder = nn.Sequential(
            nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
        )
        self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

    def encode(self, t, s):
        h = torch.randn(1, self.n_window, dtype=torch.double)
        gru_t, _ = self.gru(torch.t(t), h)
        gru_t = torch.t(gru_t)
        graph = torch.cat((t, torch.zeros(self.n_window, 1)), dim=1)
        gat_t = self.gat(torch.t(graph))
        gat_t = torch.t(gat_t)
        concat_t = torch.cat((gru_t, gat_t), dim=1)
        o, _ = self.mha(concat_t, concat_t, concat_t)
        t = self.encoder(o.view(-1)).view(self.n_hosts, self.n_latent)	
        return t

    def anomaly_decode(self, t):
        anomaly_scores = []
        for elem in t:
            anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))	
        return anomaly_scores

    def prototype_decode(self, t):
        prototypes = []
        for elem in t:
            prototypes.append(self.prototype_decoder(elem))	
        return prototypes

    def forward(self, t, s):
        t = self.encode(t, s)
        anomaly_scores = self.anomaly_decode(t)
        prototypes = self.prototype_decode(t)
        return anomaly_scores, prototypes

## Simple Multi-Head Self-Attention Model
class Attention_50(nn.Module):
    def __init__(self):
        super(Attention_50, self).__init__()
        self.name = 'Attention_50'
        self.lr = 0.0008
        self.n_hosts = 50
        self.n_feats = 3 * self.n_hosts
        self.n_window = 3 # w_size = 5
        self.n_latent = 10
        self.n_hidden = 16
        self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
        self.encoder = nn.Sequential(
            nn.Linear(self.n_window * self.n_feats, self.n_hosts * self.n_latent), nn.LeakyReLU(True),
        )
        self.anomaly_decoder = nn.Sequential(
            nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
        )
        self.prototype_decoder = nn.Sequential(
            nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
        )
        self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

    def encode(self, t, s):
        t = self.encoder(t.view(-1)).view(self.n_hosts, self.n_latent)	
        return t

    def anomaly_decode(self, t):
        anomaly_scores = []
        for elem in t:
            anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))	
        return anomaly_scores

    def prototype_decode(self, t):
        prototypes = []
        for elem in t:
            prototypes.append(self.prototype_decoder(elem))	
        return prototypes

    def forward(self, t, s):
        t = self.encode(t, s)
        anomaly_scores = self.anomaly_decode(t)
        prototypes = self.prototype_decode(t)
        return anomaly_scores, prototypes

# Generator Network : Input = Schedule, Embedding; Output = New Schedule
class Gen_50(nn.Module):
    def __init__(self):
        super(Gen_50, self).__init__()
        self.name = 'Gen_50'
        self.lr = 0.00003
        self.n_hosts = 50
        self.n_hidden = 64
        self.n = self.n_hosts * PROTO_DIM + self.n_hosts * self.n_hosts
        self.delta = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hosts * self.n_hosts), nn.Tanh(),
        )

    def forward(self, e, s):
        del_s = 4 * self.delta(torch.cat((e.view(-1), s.view(-1))))
        return s + del_s.reshape(self.n_hosts, self.n_hosts)

# Discriminator Network : Input = Schedule, New Schedule; Output = Likelihood scores
class Disc_50(nn.Module):
    def __init__(self):
        super(Disc_50, self).__init__()
        self.name = 'Disc_50'
        self.lr = 0.00003
        self.n_hosts = 50
        self.n_hidden = 64
        self.n = self.n_hosts * self.n_hosts + self.n_hosts * self.n_hosts
        self.probs = nn.Sequential(
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, 2), nn.Softmax(dim=0),
        )

    def forward(self, o, n):
        probs = self.probs(torch.cat((o.view(-1), n.view(-1))))
        return probs


############## PreGANPlus Models ##############

# 位置编码类保持不变
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 修复损失函数输入维度的Transformer_16实现
class Transformer_16(nn.Module):
    def __init__(self):
        super(Transformer_16, self).__init__()
        self.name = 'Transformer_16'
        self.lr = 0.0001
        self.n_hosts = 16  # 16个主机节点
        self.n_window = 3  # 时间窗口大小
        self.gat_input_feats = 3  # 每个主机3个特征
        self.gat_output_feats = self.n_hosts  # GAT输出维度=16
        self.d_model = self.gat_output_feats  # Transformer输入维度=16
        self.nhead = 2  # 多头注意力头数
        self.dim_feedforward = 64  # 前馈网络维度
        self.num_layers = 2  # Transformer编码器层数
        
        # 计算潜在特征的总维度（用于解码器）
        self.latent_dim = self.n_hosts * self.d_model * self.n_window  # 16*16*3=768

        # 1. 构建GAT图（与FPE_16保持一致）
        src_ids = torch.tensor([i for i in range(self.n_hosts) for _ in range(self.n_hosts)])
        dst_ids = torch.tensor([j for _ in range(self.n_hosts) for j in range(self.n_hosts)])
        self.gat_graph = dgl.graph((src_ids, dst_ids))

        # 2. GAT层（与FPE_16配置一致）
        self.gat = GAT(
            self.gat_graph,
            self.gat_input_feats,
            self.gat_output_feats
        )

        # 3. 时间编码器（输入维度为GAT输出维度16）
        self.time_encoder = nn.Linear(self.gat_output_feats, self.d_model)  # 16→16

        # 4. 位置编码器
        self.pos_encoder = PositionalEncoding(self.d_model, 0.1, self.n_window)

        # 5. Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=self.num_layers)

        # 6. 异常解码器（输出适配交叉熵损失的维度）
        self.anomaly_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 2 * self.n_hosts),  # 768→32
            nn.LeakyReLU(True),
            nn.Unflatten(1, (self.n_hosts, 2))  # 输出形状: [1, 16, 2]
        )

        # 7. 原型解码器
        self.prototype_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, PROTO_DIM * self.n_hosts),  # 768→PROTO_DIM*16
            nn.Sigmoid(),
            nn.Unflatten(1, (self.n_hosts, PROTO_DIM))  # 输出形状: [1, 16, PROTO_DIM]
        )

        # 原型初始化
        self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) 
                         for _ in range(self.n_hosts)]

    def encode(self, t, s):
        # 输入t形状：[n_window=3, n_feats=48] → 重塑为[3, 16主机, 3特征]
        t_reshaped = t.view(self.n_window, self.n_hosts, self.gat_input_feats)  # [3, 16, 3]

        # 添加batch维度适配GAT的4维输入要求 [batch=1, seq_len=3, num_nodes=16, feats=3]
        gat_input = t_reshaped.unsqueeze(0)  # [1, 3, 16, 3]

        # GAT处理：输出形状: [1, 3, 16, 16]
        gat_out = self.gat(gat_input)

        # 移除batch维度：[3, 16, 16]
        gat_out = gat_out.squeeze(0)

        # 时间编码
        time_encoded = self.time_encoder(gat_out)  # [3, 16, 16]

        # 位置编码
        pos_encoded = self.pos_encoder(time_encoded)  # [3, 16, 16]

        # Transformer编码
        memory = self.transformer_encoder(pos_encoded)  # [3, 16, 16]

        # 展平为[1, 768]（保留batch维度）
        memory_flat = memory.permute(1, 0, 2).reshape(1, -1)  # [1, 768]
        return memory_flat

    def forward(self, t, s):
        latent = self.encode(t, s)  # [1, 768]
        
        # 解码得到异常分数
        anomaly_flat = self.anomaly_decoder(latent)  # [1, 16, 2]
        
        # 关键修复：调整异常分数维度为[16, 2]（移除多余的batch维度）
        # 使输出符合交叉熵损失对输入形状[num_samples, num_classes]的要求
        anomaly_scores = [anomaly_flat[0, i].unsqueeze(0) for i in range(self.n_hosts)]  # 每个元素形状[1, 2]
        
        # 解码得到原型嵌入
        prototype_flat = self.prototype_decoder(latent)  # [1, 16, PROTO_DIM]
        prototypes = [prototype_flat[0, i] for i in range(self.n_hosts)]  # 取第0个batch
        
        return anomaly_scores, prototypes


############## Enhanced Models for PreGANPlus Enhanced ##############

# Attention-Enhanced Generator Network
class Gen_16_Attention(nn.Module):
    def __init__(self):
        super(Gen_16_Attention, self).__init__()
        self.name = 'Gen_16_Attention'
        self.lr = 0.00005
        self.n_hosts = 16
        self.n_hidden = 64
        self.proto_dim = PROTO_DIM  # 2
        
        # 1. Embedding投影层
        self.embedding_proj = nn.Linear(self.proto_dim, self.n_hidden)
        
        # 2. Schedule投影层（将16×16矩阵视为序列，每行是一个token）
        self.schedule_proj = nn.Linear(self.n_hosts, self.n_hidden)
        
        # 3. 交叉注意力：embedding作为key/value，schedule作为query
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.n_hidden,
            num_heads=4,
            batch_first=False,
            dropout=0.1
        )
        
        # 4. 自注意力：在schedule序列内部建模依赖
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.n_hidden,
            num_heads=4,
            batch_first=False,
            dropout=0.1
        )
        
        # 5. 输出层：为每个容器生成到所有主机的分配增量
        self.output = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hosts),
            nn.Tanh()
        )
    
    def forward(self, e, s):
        """
        Args:
            e: [16, 2] embedding from encoder
            s: [16, 16] current schedule matrix
        Returns:
            new_schedule: [16, 16] new schedule matrix
        """
        # 投影到相同维度
        e_proj = self.embedding_proj(e)  # [16, 64]
        s_proj = self.schedule_proj(s)   # [16, 64]
        
        # 交叉注意力：embedding指导schedule更新
        # query: schedule, key/value: embedding
        s_attended, _ = self.cross_attn(
            query=s_proj.unsqueeze(0),      # [1, 16, 64]
            key=e_proj.unsqueeze(0),        # [1, 16, 64]
            value=e_proj.unsqueeze(0)       # [1, 16, 64]
        )
        s_attended = s_attended.squeeze(0)   # [16, 64]
        
        # 自注意力：在schedule内部建模依赖
        s_self, _ = self.self_attn(
            query=s_attended.unsqueeze(0),
            key=s_attended.unsqueeze(0),
            value=s_attended.unsqueeze(0)
        )
        s_self = s_self.squeeze(0)           # [16, 64]
        
        # 残差连接
        s_fused = s_attended + s_self       # [16, 64]
        
        # 生成增量：为每个容器生成到所有主机的分配增量
        del_s = 4 * self.output(s_fused)    # [16, 16]
        
        return s + del_s


# Migration-Aware Generator Network (新设计)
class Gen_16_MigrationAware(nn.Module):
    def __init__(self):
        super(Gen_16_MigrationAware, self).__init__()
        self.name = 'Gen_16_MigrationAware'
        self.lr = 0.00005
        self.n_hosts = 16
        self.n_hidden = 64
        self.proto_dim = PROTO_DIM  # 2
        
        # 1. Embedding投影层
        self.embedding_proj = nn.Linear(self.proto_dim, self.n_hidden)
        
        # 2. Schedule投影层
        self.schedule_proj = nn.Linear(self.n_hosts, self.n_hidden)
        
        # 3. 交叉注意力：embedding作为key/value，schedule作为query
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.n_hidden,
            num_heads=4,
            batch_first=False,
            dropout=0.1
        )
        
        # 4. 自注意力：在schedule序列内部建模依赖
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.n_hidden,
            num_heads=4,
            batch_first=False,
            dropout=0.1
        )
        
        # 5. 迁移成本预测模块（新增）
        self.migration_cost_predictor = nn.Sequential(
            nn.Linear(self.n_hidden, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.ReLU()  # 迁移次数非负
        )
        
        # 6. 迁移约束层（新增）：为每个容器预测迁移概率
        self.migration_gate = nn.Sequential(
            nn.Linear(self.n_hidden, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出0-1之间的概率
        )
        
        # 7. 输出层：为每个容器生成到所有主机的分配增量
        self.output = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hosts),
            nn.Tanh()
        )
    
    def forward(self, e, s):
        """
        Args:
            e: [16, 2] embedding from encoder
            s: [16, 16] current schedule matrix
        Returns:
            new_schedule: [16, 16] new schedule matrix
            predicted_migration_cost: [1] predicted migration count
        """
        # 投影到相同维度
        e_proj = self.embedding_proj(e)  # [16, 64]
        s_proj = self.schedule_proj(s)   # [16, 64]
        
        # 交叉注意力：embedding指导schedule更新
        s_attended, _ = self.cross_attn(
            query=s_proj.unsqueeze(0),      # [1, 16, 64]
            key=e_proj.unsqueeze(0),        # [1, 16, 64]
            value=e_proj.unsqueeze(0)       # [1, 16, 64]
        )
        s_attended = s_attended.squeeze(0)   # [16, 64]
        
        # 自注意力：在schedule内部建模依赖
        s_self, _ = self.self_attn(
            query=s_attended.unsqueeze(0),
            key=s_attended.unsqueeze(0),
            value=s_attended.unsqueeze(0)
        )
        s_self = s_self.squeeze(0)           # [16, 64]
        
        # 残差连接
        s_fused = s_attended + s_self       # [16, 64]
        
        # 预测迁移成本（新增）：基于融合特征的平均值
        predicted_migration_cost = self.migration_cost_predictor(s_fused.mean(dim=0, keepdim=True))  # [1, 1]
        predicted_migration_cost = predicted_migration_cost.squeeze()  # [1]
        # 确保预测值在合理范围内（0-300），防止预测值过大导致训练不稳定
        predicted_migration_cost = torch.clamp(predicted_migration_cost, 0.0, 300.0)
        
        # 迁移门控（新增）：为每个容器预测迁移概率
        migration_gates = self.migration_gate(s_fused)  # [16, 1]
        migration_gates = migration_gates.squeeze(-1)   # [16]
        
        # 生成增量：考虑迁移约束（增强版）
        # 如果迁移概率高，大幅减小增量幅度，从而减少迁移
        del_s_raw = self.output(s_fused)  # [16, 16]
        # 进一步增强迁移约束：迁移概率高时，更大幅度减小增量（从0.8增加到0.9）
        migration_penalty = 1.0 - 0.9 * migration_gates.unsqueeze(-1)  # [16, 1]，迁移概率高时更大幅度减小增量
        # 同时考虑预测的迁移成本：如果预测迁移成本高，进一步减小增量
        # 迁移成本惩罚阈值从200.0降低到150.0，下限从0.3降低到0.2（更严格）
        migration_cost_penalty = torch.clamp(1.0 - predicted_migration_cost / 150.0, 0.2, 1.0)  # 标量
        migration_cost_penalty = migration_cost_penalty.unsqueeze(0).unsqueeze(0)  # [1, 1]，与[16, 16]广播兼容
        del_s = 4 * del_s_raw * migration_penalty * migration_cost_penalty  # [16, 16]
        
        new_schedule = s + del_s
        
        return new_schedule, predicted_migration_cost


# Multi-Task Discriminator Network
class Disc_16_MultiTask(nn.Module):
    def __init__(self):
        super(Disc_16_MultiTask, self).__init__()
        self.name = 'Disc_16_MultiTask'
        self.lr = 0.00005
        self.n_hosts = 16
        self.n_hidden = 128  # 增加隐藏层维度
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(self.n_hosts * self.n_hosts * 2, self.n_hidden * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(self.n_hidden * 2, self.n_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        
        # 任务1: 分类头（判断更好/更差）
        self.classifier = nn.Sequential(
            nn.Linear(self.n_hidden, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 2),
            nn.Softmax(dim=0)
        )
        
        # 任务2: 回归头（预测评估分数）
        self.regressor = nn.Sequential(
            nn.Linear(self.n_hidden, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)  # 输出评估分数
        )
        
        # 任务3: 响应时间预测头（用于SLA优化）
        self.response_time_predictor = nn.Sequential(
            nn.Linear(self.n_hidden, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)  # 输出预测响应时间
        )
    
    def forward(self, o, n):
        """
        Args:
            o: [16, 16] original schedule
            n: [16, 16] new schedule
        Returns:
            class_probs: [2] classification probabilities [orig_better, new_better]
            score_pred: [1] predicted evaluation score
            response_time_pred: [1] predicted response time
        """
        # 特征提取
        x = torch.cat([o.view(-1), n.view(-1)])  # [512]
        features = self.shared(x)  # [128]
        
        # 多任务输出
        class_probs = self.classifier(features)  # [2]
        score_pred = self.regressor(features)    # [1]
        response_time_pred = self.response_time_predictor(features)  # [1]
        
        return class_probs, score_pred, response_time_pred


# Multi-Objective Discriminator Network (新设计)
class Disc_16_MultiObjective(nn.Module):
    def __init__(self):
        super(Disc_16_MultiObjective, self).__init__()
        self.name = 'Disc_16_MultiObjective'
        self.lr = 0.00005
        self.n_hosts = 16
        self.n_hidden = 128
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(self.n_hosts * self.n_hosts * 2, self.n_hidden * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(self.n_hidden * 2, self.n_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        
        # 任务1: 分类头（判断更好/更差）
        self.classifier = nn.Sequential(
            nn.Linear(self.n_hidden, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 2),
            nn.Softmax(dim=0)
        )
        
        # 任务2: 能量预测头（新增）
        self.energy_predictor = nn.Sequential(
            nn.Linear(self.n_hidden, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
        
        # 任务3: 响应时间预测头
        self.response_time_predictor = nn.Sequential(
            nn.Linear(self.n_hidden, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
        
        # 任务4: 迁移成本预测头（新增）
        self.migration_cost_predictor = nn.Sequential(
            nn.Linear(self.n_hidden, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.ReLU()  # 迁移次数非负
        )
    
    def forward(self, o, n):
        """
        Args:
            o: [16, 16] original schedule
            n: [16, 16] new schedule
        Returns:
            class_probs: [2] classification probabilities
            energy_pred: [1] predicted energy consumption
            response_time_pred: [1] predicted response time
            migration_cost_pred: [1] predicted migration count
        """
        # 特征提取
        x = torch.cat([o.view(-1), n.view(-1)])  # [512]
        features = self.shared(x)  # [128]
        
        # 多目标输出
        class_probs = self.classifier(features)  # [2]
        energy_pred = self.energy_predictor(features)  # [1]
        response_time_pred = self.response_time_predictor(features)  # [1]
        migration_cost_pred = self.migration_cost_predictor(features)  # [1]
        
        return class_probs, energy_pred, response_time_pred, migration_cost_pred