import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# 生成 user-poi 邻接图
def gen_nei_graph(df, n_users, poi_id2idx_dict):
    edges = [[], [], []]  # edges[0] 是用户，edges[1] 是 POI，edges[2] 是访问频率

    # 遍历每个用户，生成用户与POI的邻接关系，并计算访问频率
    for _uid, _item in df.groupby('user_id'):
        # 使用 poi_id2idx_dict 进行 POI ID 到索引的转换
        poi_list = _item['POI_id'].tolist()
        poi_indices = [poi_id2idx_dict[poi] for poi in poi_list]  # 将 POI ID 转换为索引

        # 计算访问频率
        poi_frequency = _item['POI_id'].value_counts().to_dict()  # 计算每个POI的访问频率

        # 构建用户和POI的邻接关系
        for poi_idx in poi_indices:
            edges[0].append(_uid)  # 用户对应的行
            edges[1].append(poi_idx)  # POI 对应的列
            edges[2].append(poi_frequency[poi_list[poi_indices.index(poi_idx)]])  # 访问频率

    # 返回包含用户、POI和访问频率的邻接矩阵
    return torch.LongTensor(edges)



# 生成 POI 距离矩阵
def distance_mat_form(lat_vec, lon_vec):
    r = 6371  # 地球半径，单位为 km
    p = np.pi / 180  # 转换为弧度制
    lat_mat = np.repeat(lat_vec, lat_vec.shape[0], axis=-1)
    lon_mat = np.repeat(lon_vec, lon_vec.shape[0], axis=-1)
    a_mat = 0.5 - np.cos((lat_mat.T - lat_mat) * p) / 2 \
            + np.matmul(np.cos(lat_vec * p), np.cos(lat_vec * p).T) * (1 - np.cos((lon_mat.T - lon_mat) * p)) / 2

    dist_mat = 2 * r * np.arcsin(np.sqrt(a_mat))

    # 设置自己到自己的距离为1
    np.fill_diagonal(dist_mat, 1)

    return dist_mat

# 生成 POI 距离图
def gen_loc_graph(poi_loc, n_pois, thre):
    lat_vec = np.array([poi_loc[_poi][0] for _poi in range(n_pois)], dtype=np.float64).reshape(-1, 1)
    lon_vec = np.array([poi_loc[_poi][1] for _poi in range(n_pois)], dtype=np.float64).reshape(-1, 1)
    dist_mat = distance_mat_form(lat_vec, lon_vec)  # 生成距离矩阵

    # 使用给定的距离阈值生成邻接矩阵
    adj_mat = np.triu(dist_mat <= thre, k=1)
    num_edges = adj_mat.sum()

    # 获取邻接矩阵中的边
    idx_mat = np.arange(n_pois).reshape(-1, 1).repeat(n_pois, -1)
    row_idx = idx_mat[adj_mat]
    col_idx = idx_mat.T[adj_mat]
    edges = np.stack((row_idx, col_idx))

    # 构建 POI 的邻接字典
    nei_dict = {poi: [] for poi in range(n_pois)}
    for e_idx in range(edges.shape[1]):
        src, dst = edges[:, e_idx]
        nei_dict[src].append(dst)
        nei_dict[dst].append(src)

    # 返回距离矩阵、边信息和邻接字典
    return dist_mat, edges, nei_dict

class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W.to(X.device))
        e = self._prepare_attentional_mechanism_input(Wh)
        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :].to(Wh.device))
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :].to(Wh.device))
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)



class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x

class GCNWithDistMatrix(nn.Module):
    def __init__(self, in_features, nhid, noutput, adj_matrix, dist_matrix, dropout):
        super(GCNWithDistMatrix, self).__init__()
        self.gcn = GCN(in_features, nhid, noutput, dropout)
        self.adj_matrix = adj_matrix
        self.dist_matrix = dist_matrix

        # 将距离矩阵的值反转为权重：距离越远权重越小
        # 添加一个小的常数来避免除以 0 的情况
        self.reversed_dist_matrix = 1 / (dist_matrix + 1e-6)

        # 对反转后的距离矩阵进行归一化处理
        self.normalized_dist_matrix = self.reversed_dist_matrix / torch.max(self.reversed_dist_matrix)

    def forward(self, X, A):
        # 将邻接矩阵与归一化后的反转距离矩阵相乘
        adjusted_adj_matrix = A * self.normalized_dist_matrix

        # 通过GCN进行传播
        return self.gcn(X, adjusted_adj_matrix)

class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed

class POIEmbeddings(nn.Module):
    def __init__(self, num_pois, embedding_dim):
        super(POIEmbeddings, self).__init__()
        self.poi_embedding = nn.Embedding(num_embeddings=num_pois, embedding_dim=embedding_dim)

    def forward(self, poi_idx):
        embed = self.poi_embedding(poi_idx)
        return embed
        
class AdjEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim, user_poi_edges):
        super(AdjEmbeddings, self).__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.user_poi_edges = user_poi_edges  # 包含访问频率的用户-POI邻接关系
        self.fc = nn.Linear(embedding_dim + embedding_dim, embedding_dim)  # 拼接后的线性层

    def forward(self, user_idx, poi_embeddings):
        # 获取用户嵌入
        user_embed = self.user_embedding(user_idx)

        # 获取访问频率
        user_poi_mask = self.user_poi_edges[0] == user_idx  # 找到与用户相关的访问记录
        user_poi_indices = self.user_poi_edges[1][user_poi_mask]  # 获取该用户访问的POI索引
        visit_frequencies = self.user_poi_edges[2][user_poi_mask].float()  # 获取用户对每个POI的访问频率

        poi_neighbor_embeds = poi_embeddings[user_poi_indices]

        # 根据访问频率计算加权平均嵌入
        if len(poi_neighbor_embeds) > 0:
            weights = visit_frequencies.unsqueeze(1)  # 权重为访问频率，扩展为二维张量
            weighted_poi_neighbors_embed = (poi_neighbor_embeds * weights).sum(dim=0) / weights.sum(dim=0)
            weighted_poi_neighbors_embed = weighted_poi_neighbors_embed.unsqueeze(0)  # [1, embedding_dim]
        else:
            weighted_poi_neighbors_embed = torch.zeros_like(user_embed)

        # 拼接嵌入
        combined_embed = torch.cat((user_embed, weighted_poi_neighbors_embed), dim=1)

        combined_embed = self.fc(combined_embed)
        return combined_embed



class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x

# 特征融合（共享底层）
class Poi_representation(nn.Module):
    def __init__(self, poi_emb_dim, cat_emb_dim):
        super(Poi_representation, self).__init__()
        self.total_emb_dim = poi_emb_dim + cat_emb_dim
        self.conv1 = nn.Conv1d(in_channels=self.total_emb_dim, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)

        # 计算卷积输出的大小
        self.fc = nn.Linear(64, self.total_emb_dim)

    def forward(self, poi_embedding, cat_embedding):
        embeddings = torch.cat((poi_embedding, cat_embedding), dim=-1).unsqueeze(0).unsqueeze(2)
        x = F.relu(self.conv1(embeddings))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.squeeze(0)

# MLP交叉学习
class FuseBlock(nn.Module):
    def __init__(self, dim_1, dim_2):
        super(FuseBlock, self).__init__()
        embed_dim = dim_1 + dim_2
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, dim_1, dim_2):
        x = self.fuse_embed(torch.cat((dim_1, dim_2), 0))
        x = self.leaky_relu(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Linear(d_ffn, d_ffn)  # 用 nn.Linear 替代 nn.Conv1d
        nn.init.constant_(self.spatial_proj.bias, 1.0)
        nn.init.xavier_normal_(self.spatial_proj.weight)  # 初始化线性层的权重

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)  # 将输入拆成两部分
        v = self.norm(v)
        v = self.spatial_proj(v)  # 线性投影，处理动态长度的序列
        out = u * v  # 门控操作
        return out

class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)  # 维度调整，2倍 FFN 尺寸
        self.channel_proj2 = nn.Linear(d_ffn, d_model)  # 输出维度与输入保持一致
        self.sgu = SpatialGatingUnit(d_ffn)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual  # 残差连接
        return out

class gMLP(nn.Module):
    def __init__(self, d_model=320, d_ffn=256, num_layers=2):
        super().__init__()
        self.model = nn.Sequential(*[gMLPBlock(d_model, d_ffn) for _ in range(num_layers)])

    def forward(self, x):
        return self.model(x)

class Task1Decoder(nn.Module):
    def __init__(self, d_model, num_poi):
        super().__init__()

        self.dropout = nn.Dropout(0.3)
        self.temperature = d_model ** 0.5  # 温度缩放
        # 基础隐式特征
        self.implicit_base_features = nn.Linear(d_model, num_poi, bias=False)
        nn.init.xavier_normal_(self.implicit_base_features.weight)

        # 隐式注意力特征
        self.implicit_att_features = nn.Linear(d_model, num_poi, bias=False)
        nn.init.xavier_normal_(self.implicit_att_features.weight)

    def forward(self, enc_output):
        # 基础隐式特征
        base_implicit = self.implicit_base_features(enc_output)
        base_implicit = F.normalize(base_implicit, p=2, dim=-1, eps=1e-05)

        # 注意力隐式特征
        attn = torch.matmul(enc_output / self.temperature, enc_output.transpose(1, 2))  # 自注意力机制
        attn = self.dropout(torch.tanh(attn))
        seq1_implicit = torch.matmul(attn, enc_output)
        seq1_implicit = self.implicit_att_features(seq1_implicit)
        seq1_implicit = F.normalize(seq1_implicit, p=2, dim=-1, eps=1e-05)

        # 汇总所有特征
        output = base_implicit + seq1_implicit
        output = torch.tanh(output)
        return output

class Task1Model(nn.Module):
    def __init__(self, num_poi, d_model=320, d_ffn=256, num_layers=2):
        super(Task1Model, self).__init__()
        self.gmlp = gMLP(d_model=d_model, d_ffn=d_ffn, num_layers=num_layers)
        self.decoder = Task1Decoder(d_model, num_poi)

    def forward(self, embeddings):
        """
        输入: embeddings [batch_size, seq_len, dim]
        输出: [batch_size, num_poi]
        """
        # 使用 gMLP 进行编码
        enc_output = self.gmlp(embeddings)

        # 解码得到 POI 的预测值 [batch_size, seq_len, num_poi]
        poi_predictions = self.decoder(enc_output)

        # 沿 seq_len 维度求和聚合 [batch_size, num_poi]
        aggregated_predictions = poi_predictions.mean(dim=1)

        return aggregated_predictions  # 返回最终的 POI 推荐预测值

class Pretask1(nn.Module):
    def __init__(self, num_pois, total_dim):
        super(Pretask1, self).__init__()
        self.poi_output_layer = nn.Sequential(
            nn.Linear(total_dim, 512),  # 增加一个中间层
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, num_pois)
        )
    def forward(self, x):
        #print('x.shape: ', x.shape)
        x = self.poi_output_layer(x)
        x = x.mean(dim=1)
        return x