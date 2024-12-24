#!/usr/bin/env python
# coding=utf-8
"""
@Description   嵌入优化的 LightGCN 推荐
@Author        Alex_McAvoy
@Date          2024-10-14 15:15:05
"""

import numpy as np
import pandas as pd
import torch
import ast

from const import cfg
from utils.log import logger
from utils.wrapper import calTimes
from model.LightGCNOpti.model import LightGCNOpti
from model.LightGCNOpti.train import trainLightGCNOpti
from utils.graph import convertEdgeIndexToAdjMatrix, convertAdjMatrixToEdgeIndex
from utils.trans import getUserItemsDictByEdgeIndex

@calTimes(logger, "LightGCNOpti图建立完成")
def buildGraph(user_num:int, item_num: int, rating_df: pd.DataFrame, 
               train_data_df: pd.DataFrame, val_data_df: pd.DataFrame, 
               test_data_df: pd.DataFrame) -> tuple:
    """
    @description 构建LightGCN图
    @param {int} user_num 总用户数
    @param {int} item_num 总项目数
    @param {pd} rating_df 过滤后的评分数据
    @param {pd} train_data_df 训练数据
    @param {pd} val_data_df 验证数据
    @param {pd} test_data_df 测试数据
    @return {tuple} 元组包含
            - {torch} edge_index 所有的边索引
            - {torch} train_edge_index 训练邻接矩阵
            - {torch} val_edge_index 验证邻接矩阵
            - {torch} test_edge_index 测试邻接矩阵
    """

    # 所有的边索引
    users_node = torch.tensor(rating_df["user_id"].values, dtype=torch.long)
    items_node = torch.tensor(rating_df["item_id"].values, dtype=torch.long)
    edge_index = torch.stack([users_node, items_node], dim=0)

    # 训练集边索引
    train_users_node = torch.tensor(train_data_df["user_id"].values, dtype=torch.long)
    train_items_node = torch.tensor(train_data_df["item_id"].values, dtype=torch.long)
    train_edge_index = torch.stack([train_users_node, train_items_node], dim=0)

    # 验证集边索引
    val_users_node = torch.tensor(val_data_df["user_id"].values, dtype=torch.long)
    val_items_node = torch.tensor(val_data_df["item_id"].values, dtype=torch.long)
    val_edge_index = torch.stack([val_users_node, val_items_node], dim=0)

    # 测试集边索引
    test_users_node = torch.tensor(test_data_df["user_id"].values, dtype=torch.long)
    test_items_node = torch.tensor(test_data_df["item_id"].values, dtype=torch.long)
    test_edge_index = torch.stack([test_users_node, test_items_node], dim=0)
    
    # 转为邻接矩阵
    train_edge_index = convertEdgeIndexToAdjMatrix(user_num, item_num, train_edge_index)
    val_edge_index = convertEdgeIndexToAdjMatrix(user_num, item_num, val_edge_index)
    test_edge_index = convertEdgeIndexToAdjMatrix(user_num, item_num, test_edge_index)

    return edge_index, train_edge_index, val_edge_index, test_edge_index

def recommendForAllUser(model: LightGCNOpti, user_num: int, item_num: int,
                        train_edge_index: torch.Tensor, val_edge_index: torch.Tensor, 
                        test_edge_index: torch.Tensor, k: int) -> dict:
    """
    @description 获取所有用户的推荐字典
    @param {LightGCNOpti} model 嵌入优化的LightGCN模型
    @param {int} user_num 总用户数
    @param {int} item_num 总项目数
    @param {pd} train_edge_index 训练邻接矩阵
    @param {pd} val_edge_index 验证邻接矩阵
    @param {pd} test_edge_index 测试邻接矩阵
    @param {int} k 推荐列表大小
    @return {dict} 所有用户的推荐字典
    """
    # 获取Embedding向量
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight
    # 计算用户对项目的偏好矩阵，shape(user_num, item_num)
    score = torch.matmul(user_embedding, item_embedding.T)

    # 从邻接矩阵转为边索引
    train_edge_index = convertAdjMatrixToEdgeIndex(user_num, item_num, train_edge_index)
    val_edge_index = convertAdjMatrixToEdgeIndex(user_num, item_num, val_edge_index)
    test_edge_index = convertAdjMatrixToEdgeIndex(user_num, item_num, test_edge_index)

    # 排除训练集中已知的正样本
    exclude_users = []
    exclude_items = []
    user_pos_items = getUserItemsDictByEdgeIndex(train_edge_index)
    for user, items in user_pos_items.items():
        exclude_users.extend([user] * len(items))
        exclude_items.extend(items)
    # 对于训练集中已知的用户-项目对，将他们的评分设为一个负值，以确保不会出现在推荐列表中
    score[exclude_users, exclude_items] = -(1 << 10) 

    # 排除验证集中已知的正样本
    exclude_users = []
    exclude_items = []
    user_pos_items = getUserItemsDictByEdgeIndex(val_edge_index)
    for user, items in user_pos_items.items():
        exclude_users.extend([user] * len(items))
        exclude_items.extend(items)
    # 对于验证集中已知的用户-项目对，将他们的评分设为一个负值，以确保不会出现在推荐列表中
    score[exclude_users, exclude_items] = -(1 << 10)

    # 获取topK推荐
    _, recommendations = torch.topk(score, k=k)

    # 转为字典
    all_user_recommend_dict = {}
    for uid, items in enumerate(recommendations):
        all_user_recommend_dict[uid] = items.tolist()

    # 存储推荐结果
    np.save(cfg.RECOMMEND["save_path"] + "all_user_recommend_dict_" + cfg.MODEL["name"] + str(cfg.RECOMMEND["k"]) + ".npy", all_user_recommend_dict)

    
    return all_user_recommend_dict

def recommendLightGCNOpti(user_num:int, item_num: int, 
                      rating_df: pd.DataFrame, train_data_df: pd.DataFrame, 
                      val_data_df: pd.DataFrame, test_data_df: pd.DataFrame, 
                      user_features_df: pd.DataFrame, item_features_df: pd.DataFrame) -> dict:
    """
    @description LightGCN推荐
    @param {int} user_num 总用户数
    @param {int} item_num 总项目数
    @param {pd} rating_df 过滤后的评分数据
    @param {pd} train_data_df 训练数据
    @param {pd} val_data_df 验证数据
    @param {pd} test_data_df 测试数据
    @param {pd} user_features_df 用户特征
    @param {pd} item_features_df 项目特征
    @return {dict} 所有用户的推荐字典
    """

    # 超参
    k = cfg.RECOMMEND["k"] # 推荐列表大小

    # 构建LightGCN图
    edge_index, train_edge_index, val_edge_index, test_edge_index = buildGraph(user_num, item_num, rating_df, train_data_df, val_data_df, test_data_df)

    # 用户特征Dataframe转为tensor
    user_features_df_sorted = user_features_df.sort_values(by="user_id")
    user_features_np = user_features_df_sorted["user_features"].apply(
        lambda row: ast.literal_eval(row) if not isinstance(row, list) else row
    ).tolist()
    user_features_np = np.array(user_features_np)
    user_features = torch.from_numpy(user_features_np).float()
    # 项目特征Dataframe转为tensor
    item_features_df_sorted = item_features_df.sort_values(by="item_id")
    item_features_np = item_features_df_sorted["item_features"].apply(
        lambda row: ast.literal_eval(row) if not isinstance(row, list) else row
    ).tolist()
    item_features_np = np.array(item_features_np)
    item_features = torch.from_numpy(item_features_np).float()
    
    # 训练
    logger.info("正在加载模型")
    try:   
        model = torch.load(cfg.MODEL["save_path"] + str(cfg.RECOMMEND["k"]) + "_LightGCNOpti.pth")
        logger.info("模型加载完毕")
    except:
        logger.info("模型加载失败，正在重新训练模型")
        model = trainLightGCNOpti(user_num, item_num, edge_index, train_edge_index, val_edge_index, user_features, item_features)
   
    # 获取推荐字典
    all_user_recommend_dict = recommendForAllUser(model, user_num, item_num, train_edge_index, val_edge_index, test_edge_index, k)

    return all_user_recommend_dict
