#!/usr/bin/env python
# coding=utf-8
"""
@Description   扩散+嵌入优化的LightGCN模型
@Author        Alex_McAvoy
@Date          2024-10-17 17:59:54
"""

import numpy as np
import pandas as pd
import torch

from const import cfg
from utils.log import logger
from utils.wrapper import calTimes
from model.LightGCN.model import LightGCN
from utils.graph import convertAdjMatrixToEdgeIndex
from utils.trans import getUserItemsDictByEdgeIndex
from model.LightGCN.recommend import buildGraph
from model.LightGCN.train import trainLightGCN
from model.SpreadMethod.model import getSpreadingGeneralMat, HybridS, getResource
from utils.trans import getInteractionMatrixByDataframe

def getLightGCNModel(user_num:int, item_num: int, rating_df: pd.DataFrame, 
                     train_data_df: pd.DataFrame, val_data_df: pd.DataFrame, 
                     test_data_df: pd.DataFrame, k: int) -> tuple:
    """
    @description 获取LightGCN模型
    @param {int} user_num 总用户数
    @param {int} item_num 总项目数
    @param {pd} rating_df 过滤后的评分数据
    @param {pd} train_data_df 训练数据
    @param {pd} val_data_df 验证数据
    @param {pd} test_data_df 测试数据
    @return {tuple} 元组包含
            - {LightGCN} LightGCN 模型
            - {torch} edge_index 所有的边索引
            - {torch} train_edge_index 训练邻接矩阵
            - {torch} val_edge_index 验证邻接矩阵
            - {torch} test_edge_index 测试邻接矩阵
    """
    # 构建LightGCN图
    edge_index, train_edge_index, val_edge_index, test_edge_index = buildGraph(user_num, item_num, rating_df, train_data_df, val_data_df, test_data_df)
    
    # 训练
    logger.info("正在加载LightGCN模型")
    try:
        model = torch.load(cfg.MODEL["save_path"] + str(k) + "_LightGCN.pth")
        logger.info("LightGCN模型加载完毕")
    except:
        logger.info("LightGCN模型加载失败，正在重新训练模型")
        model = trainLightGCN(user_num, item_num, edge_index, train_edge_index, val_edge_index)
    return model, edge_index, train_edge_index, val_edge_index, test_edge_index

@calTimes(logger, "分配权重矩阵计算完成")
def getAllocateMat(user_num:int, item_num: int, rating_df: pd.DataFrame, 
                   train_data_df: pd.DataFrame, val_data_df: pd.DataFrame, 
                   test_data_df: pd.DataFrame, k: int) -> np.ndarray:
    """
    @description 计算分配权重矩阵
    @param {LightGCN} model LightGCN模型
    @param {int} user_num 总用户数
    @param {int} item_num 总项目数
    @param {pd} train_edge_index 训练邻接矩阵
    @param {pd} val_edge_index 验证邻接矩阵
    @param {pd} test_edge_index 测试邻接矩阵
    @return {np} 分配权重矩阵
    """

    # 获取LightGCN模型
    model, edge_index, train_edge_index, val_edge_index, test_edge_index = getLightGCNModel(user_num, item_num, rating_df, train_data_df, val_data_df, test_data_df, k)
            
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

    return score.detach().cpu().numpy()

@calTimes(logger, "资源扩散矩阵计算完成")
def getHybridSResourceMat(A: np.ndarray, general_W: np.ndarray, lambad_val: float) -> np.ndarray:
    """
    @description 获取资源分配矩阵
    @param {np} A 用户项目交互矩阵
    @param {np} general_W 通用扩散矩阵
    @param {float} lambda_val HybridS 混合比例常数
    @return {np} 资源分配矩阵 
    """
    # 获取转移矩阵
    W = HybridS(A, general_W, lambad_val)
    # 获取扩散后的资源值
    F_new = getResource(A, W)

    return F_new

def getResourceMat(user_num:int, item_num: int, rating_df: pd.DataFrame, 
                   train_data_df: pd.DataFrame, val_data_df: pd.DataFrame, 
                   test_data_df: pd.DataFrame) -> np.ndarray:
    """
    @description 计算最终的资源矩阵
    @param {int} user_num 总用户数
    @param {int} item_num 总项目数
    @param {pd} rating_df 过滤后的评分数据
    @param {pd} train_data_df 训练数据
    @param {pd} val_data_df 验证数据
    @param {pd} test_data_df 测试数据
    @return {np} 资源矩阵
    """
    # 超参
    k = cfg.RECOMMEND["k"]                              # 推荐列表大小
    lambda_val = cfg.MODEL["HyperParameter"]["lambda"]  # HybridS 混合比例常数

    # 根据LightGCN模型，获取分配权重矩阵
    G = getAllocateMat(user_num, item_num, rating_df, train_data_df, val_data_df, test_data_df, k)

    # 根据训练集和测试集，获取用户-项目交互矩阵
    A = getInteractionMatrixByDataframe(user_num, item_num, pd.concat([train_data_df, val_data_df]))

    # 计算通用扩散矩阵
    general_W = getSpreadingGeneralMat(A)
    # 根据HybridS模型，获取资源扩散矩阵
    F = getHybridSResourceMat(A, general_W, lambda_val)

    # 计算二者的哈达玛积
    F_new = G * F

    return F_new
