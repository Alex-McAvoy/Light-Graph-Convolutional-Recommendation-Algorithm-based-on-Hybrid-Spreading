#!/usr/bin/env python
# coding=utf-8
"""
@Description   扩散方法模型
@Author        Alex_McAvoy
@Date          2024-10-17 20:05:51
"""
import numpy as np

from utils.log import logger
from utils.wrapper import calTimes

@calTimes(logger, "通用扩散矩阵计算完成")
def getSpreadingGeneralMat(A: np.ndarray) -> np.ndarray:
    """
    @description 获取通用扩散矩阵
    @param {np} A 用户-物品交互矩阵
    @return {np} 通用扩散矩阵，用于后续物质扩散、热扩散、混合扩散
    """
    # 计算用户的度：每个用户交互过多少物品
    user_degrees = np.sum(A, axis=1)
    # 处理用户度数为零的情况，避免除以零
    user_degrees[user_degrees == 0] = 1
    # 计算通用转移矩阵
    general_W = np.dot(A.T / user_degrees, A)

    return general_W

@calTimes(logger, "扩散资源矩阵计算完成")
def ProbS(A: np.ndarray, general_W: np.ndarray) -> np.ndarray:
    """
    @description 概率扩散
    @param {np} A 用户-物品交互矩阵
    @param {np} general_W 通用转移矩阵
    @return {np} 转移矩阵，W[i][j]为物品i到物品j的转移值
    """
    # 计算物品的度：每个物品被多少用户交互
    item_degrees = np.sum(A, axis=0)
    # 处理物品度数为零的情况，避免除以零
    item_degrees[item_degrees == 0] = 1
    # 计算转移矩阵
    W = general_W / item_degrees[np.newaxis, :]
    return W

@calTimes(logger, "扩散资源矩阵计算完成")
def HeatS(A: np.ndarray, general_W: np.ndarray) -> np.ndarray:
    """
    @description 热扩散
    @param {np} A 用户-物品交互矩阵
    @param {np} general_W 通用转移矩阵
    @return {np} 转移矩阵，W[i][j]为物品i到物品j的转移值
    """
    # 计算物品的度：每个物品被多少用户交互
    item_degrees = np.sum(A, axis=0)
    # 处理物品度数为零的情况，避免除以零
    item_degrees[item_degrees == 0] = 1
    # 计算转移矩阵
    W = general_W / item_degrees[:, np.newaxis]

    return W

@calTimes(logger, "扩散资源矩阵计算完成")
def HybridS(A: np.ndarray, general_W: np.ndarray, Lambda: float) -> np.ndarray:
    """
    @description 概率扩散与热扩散混合推荐
    @param {np} A 用户-物品交互矩阵
    @param {np} general_W 通用转移矩阵
    @param {float} Lambda 混合算法比例常数，Lambda=0时，Hybrid退化为HeatS，Lambda=1时，Hybrid退化为ProbS
    @return {np} 转移矩阵，W[i][j]为物品i到物品j的转移值
    """
    # 计算物品的度：每个物品被多少用户交互
    item_degrees = np.sum(A, axis=0)

    # 计算物品度的混合幂次
    degree_alpha = np.power(item_degrees, 1 - Lambda)
    degree_beta = np.power(item_degrees, Lambda)
    item_degrees = degree_alpha[:, np.newaxis] * degree_beta[np.newaxis, :]

    # 处理物品度数为零的情况，避免除以零
    item_degrees[item_degrees == 0] = 1
    
    # 计算转移矩阵
    W = general_W / item_degrees

    return W

@calTimes(logger, "资源矩阵计算完成")
def getResource(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    @description 获取扩散后的目标用户对应的物品资源值
    @param {np} A 用户-物品交互矩阵
    @param {np} W 转移矩阵
    @return {np} 扩散后的物品资源值，f_new[i]为用户i经过两轮扩散后的各物品的资源向量
    """
    # 初始资源向量：已交互过为1，未交互过为0
    F0 = A
    # 扩散
    F_new = np.dot(F0, W)
    return F_new
