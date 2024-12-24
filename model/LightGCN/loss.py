#!/usr/bin/env python
# coding=utf-8
"""
@Description   BPR损失
@Author        Alex_McAvoy
@Date          2024-10-15 13:04:34
"""
import random
import torch
from torch_geometric.utils import structured_negative_sampling

def BPRLoss(users_emb_final: torch.Tensor, users_emb_0: torch.Tensor, 
            pos_items_emb_final: torch.Tensor, pos_items_emb_0: torch.Tensor, 
            neg_items_emb_final: torch.Tensor, neg_items_emb_0: torch.Tensor, 
            lambda_val: float) -> float:
    """
    @description BPR损失函数
    @param {torch} users_emb_final 用户最终Embedding向量
    @param {torch} users_emb_0 用户初始Embedding
    @param {torch} pos_items_emb_final 正样本最终Embedding向量
    @param {torch} pos_items_emb_0 正样本初始Embedding向量
    @param {torch} neg_items_emb_final 负样本最终Embedding向量
    @param {torch} neg_items_emb_0 负样本初始Embedding向量
    @param {float} lambda_val 超参，控制L2正则化强度
    @return {float} BPR损失
    """
    
    # L2 损失
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) + pos_items_emb_0.norm(2).pow(2) + neg_items_emb_0.norm(2).pow(2))

    # 正样本的预测分数
    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1) 
    # 负样本的预测分数
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)

    # BPR损失
    bpr_loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores))
    
    # 总损失
    loss = bpr_loss + reg_loss
    
    return loss

def sampleMiniBatch(batch_size: int, edge_index: torch.Tensor) -> tuple:
    """
    @description 计算BPR损失的辅助函数，由于LightGCN是自监督学习，依赖于图结构本身，因此需要该函数随机采样一小批正负样本
    @param {int} batch_size 批大小
    @param {torch} edge_index 边索引
    @return {tuple} 元组包含
            - {torch} user_indices 用户索引
            - {torch} pos_item_indices 正样本索引
            - {torch} neg_item_indices 负样本索引
    """

    # 从边索引中负采样，返回值为三元组(users, pos_items, neg_items)，分别代表用户索引、正样本索引、负样本索引
    edges = structured_negative_sampling(edge_index)
    
    # 将采样的索引堆叠为2D张量，shape变为 (3, 边数)，每一行分别代表用户索引、正样本索引、负样本索引
    edges = torch.stack(edges, dim=0)
    
    # 在负采样的索引中，随机采样batch_size个索引，构建mini-batch
    indices = random.choices([i for i in range(edges[0].shape[0])], k=batch_size)
    
    # 根据随机采样结果，从edges中选取对应batch数据，shape(3, batch_size)
    batch = edges[:, indices]
    user_indices, pos_item_indices, neg_item_indices = batch[0], batch[1], batch[2]

    return user_indices, pos_item_indices, neg_item_indices
