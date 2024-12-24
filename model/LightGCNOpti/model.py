#!/usr/bin/env python
# coding=utf-8
"""
@Description   嵌入优化后的LightGCN模型
@Author        Alex_McAvoy
@Date          2024-10-15 12:59:46
"""
import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class LightGCNOpti(MessagePassing):
    def __init__(self, user_num: int, item_num: int, 
                 embedding_dim: int, layers: int, 
                 user_features: torch.Tensor, item_features: torch.Tensor) -> None:
        """
        @description LightGCN模型
        @param {*} self 类实例化对象
        @param {int} user_num 用户数
        @param {int} item_num 项目数
        @param {int} embedding_dim Embedding维度数
        @param {int} layers GCN层数
        @param {torch} user_features 用户特征
        @param {torch} item_features 项目特征
        """
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.layers = layers

        # 定义用于将用户和项目特征映射到 embedding_dim 维度的线性层
        self.user_linear = nn.Linear(user_features.size(1), embedding_dim)
        self.item_linear = nn.Linear(item_features.size(1), embedding_dim)

        # 处理特征并将其用作初始权重
        user_emb_init = self.user_linear(user_features)
        item_emb_init = self.item_linear(item_features)

        # 用户Embedding层，初始权重设为 user_emb_init，shape： (user_num, embedding_dim)
        self.users_emb = nn.Embedding(num_embeddings=self.user_num, embedding_dim=self.embedding_dim)
        self.users_emb.weight = nn.Parameter(user_emb_init)

        # 项目Embedding层，初始权重设为 item_emb_init，shape： (item_num, embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.item_num, embedding_dim=self.embedding_dim)
        self.items_emb.weight = nn.Parameter(item_emb_init)


    def forward(self, edge_index: torch.Tensor) -> tuple:
        """
        @description 前向传播
        @param {*} self 类实例化对象
        @param {torch} edge_index 邻接矩阵
        @return {tuple} 元组包含
                - {torch} e_u^k 第k层用户Embedding
                - {torch} e_u^0 初始用户Embedding
                - {torch} e_i^k 第k层项目Embedding
                - {torch} e_i^0 初始项目Embedding
        """
        
        # 归一化处理
        edge_index_norm = gcn_norm(edge_index=edge_index, add_self_loops=False)

        # 初始Embedding向量 e^0，用户Embedding向量和项目Embedding向量的连接，shape(user_num+item_num, embedding_dim)
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        embs = [emb_0]
        
        # 进行layers层传播
        emb_k = emb_0 
        for i in range(self.layers):
            emb_k = self.propagate(edge_index=edge_index_norm[0], x=emb_k, norm=edge_index_norm[1])
            embs.append(emb_k)
            
        # 堆叠每一层的Embedding矩阵，shape(user_num+item_num , n_layers+1 ,embedding_dim)
        embs = torch.stack(embs, dim=1)
        
        # 将a_k统一设成 1/(layers+1)，即求平均
        emb_final = torch.mean(embs, dim=1) # e^k

        # 分割最终的用户Embedding向量 e_u^k 和项目Embedding向量 e_i^k
        users_emb_final, items_emb_final = torch.split(emb_final, [self.user_num, self.item_num]) 

        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j, norm) -> torch.Tensor:
        """
        @description 消息传递
        @param {*} self 类实例化对象
        @param {torch} x_j 源点在边索引中的所有邻居，shape: (边数, embedding_dim)
        @param {torch} norm 归一化参数，shape: (边数)
        @return {torch} 每条边特征的加权矩阵
        """
        return norm.view(-1, 1) * x_j

