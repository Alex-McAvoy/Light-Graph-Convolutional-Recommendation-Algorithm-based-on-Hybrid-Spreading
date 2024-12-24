#!/usr/bin/env python
# coding=utf-8
"""
@Description   嵌入优化的LightGCNO模型评估
@Author        Alex_McAvoy
@Date          2024-10-15 13:06:52
"""
import numpy as np
import torch
from torch_geometric.utils import structured_negative_sampling

from model.LightGCNOpti.model import LightGCNOpti
from model.LightGCNOpti.loss import BPRLoss
from utils.trans import getUserItemsDictByEdgeIndex
from utils.graph import convertAdjMatrixToEdgeIndex

def getValRecommendations(model: LightGCNOpti, user_num: int, item_num: int, 
                            train_edge_index: torch.Tensor, val_edge_index: torch.Tensor, 
                            k: int) -> torch.Tensor:
    """
    @description 获取验证阶段的用户推荐
    @param {LightGCN} model LightGCN模型 
    @param {int} user_num 用户数
    @param {int} item_num 项目数
    @param {torch} train_edge_index 训练数据邻接矩阵
    @param {torch} val_edge_index 验证数据邻接矩阵
    @param {int} k 推荐列表大小
    @return {torch} 用户推荐列表，Recommendations[i]为用户i的推荐列表
    """
    # 获取用户、项目Embedding向量
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight
    # 计算用户对项目的偏好矩阵，shape(user_num, item_num)
    score = torch.matmul(user_embedding, item_embedding.T)

    # 从邻接矩阵转为边索引
    train_edge_index = convertAdjMatrixToEdgeIndex(user_num, item_num, train_edge_index)
    val_edge_index = convertAdjMatrixToEdgeIndex(user_num, item_num, val_edge_index)

    # 排除训练集中已知的正样本
    exclude_users = []
    exclude_items = []
    user_pos_items = getUserItemsDictByEdgeIndex(train_edge_index)
    for user, items in user_pos_items.items():
        exclude_users.extend([user] * len(items))
        exclude_items.extend(items)

    # 对于训练集中已知的用户-项目对，将他们的评分设为一个负值，以确保不会出现在推荐列表中
    score[exclude_users, exclude_items] = -(1 << 10) 

    # 获取topK推荐
    _, recommendations = torch.topk(score, k=k)

    return recommendations

def calValLoss(model: LightGCNOpti, user_num: int, item_num: int, val_edge_index: torch.Tensor, lambda_val: float) -> float:
    """
    @description 计算训练阶段的验证损失
    @param {LightGCN} model LightGCN模型
    @param {int} user_num 用户数
    @param {int} item_num 项目数
    @param {torch} val_edge_index 验证数据邻接矩阵
    @param {float} lambda_val 控制BPR损失L2正则化强度
    @return {float} 验证损失
    """

    # 获取验证数据，从模型输出的用户、项目的，最终、初始的Embedding向量
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(val_edge_index)
    
    # 从验证数据中负采样，返回用户索引、正样本索引、负样本索引三元组(users, pos_items, neg_items)
    r_mat_edge_index = convertAdjMatrixToEdgeIndex(user_num, item_num, val_edge_index)
    edges = structured_negative_sampling(r_mat_edge_index, contains_neg_self_loops=False)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    
    # 从验证数据中，采样得到的用户、正样本、负样本，所对应的用户最终、初始Embedding向量
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]

    # 计算验证数据的BPR损失
    valLoss = BPRLoss(users_emb_final, users_emb_0, 
                   pos_items_emb_final, pos_items_emb_0, 
                   neg_items_emb_final, neg_items_emb_0, 
                   lambda_val).item()
    
    return round(valLoss, 5)
