#!/usr/bin/env python
# coding=utf-8
"""
@Description   计算多样性指标
@Author        Alex_McAvoy
@Date          2024-10-17 22:09:56
"""
import numpy as np
import torch

from utils.log import logger
from utils.wrapper import calTimes


def calHammingDistance(recommendations: torch.Tensor, k: int) -> float:
    """
    @description 计算海明距离
    @param {torch} recommendations 所有用户的TopK推荐
    @param {int} k 推荐列表大小
    @return {float} hamming_distance 海明距离
    """

    # 用户数
    user_num, item_num = recommendations.shape

    # 总海明距离
    total_H = 0.0

    # 记忆数据，记录两个用户的推荐列表的重复值
    memory_data = {}

    for uid_i in range(user_num):
        for uid_j in range(user_num):
            # 跳过相同用户
            if uid_i == uid_j:
                continue
            if uid_i < uid_j:
                key = f"{uid_i}-{uid_j}"
            else:
                key = f"{uid_j}-{uid_i}"

            if key in memory_data:
                hamming = memory_data[key]
            else:
                # 用户i的推荐列表
                recommend_iid_i = set(recommendations[uid_i].tolist())

                # 用户j的推荐列表
                recommend_iid_j = set(recommendations[uid_j].tolist())

                # 计算用户i、用户j的推荐项目重叠数
                Q = len(recommend_iid_i & recommend_iid_j)

                # 计算用户i、用户j的海明距离
                hamming = 1 - (Q / k)
                memory_data[key] = hamming

            total_H += hamming

    # 平均海明距离
    H = round(total_H / (user_num * (user_num - 1)), 5)

    return round(H, 5)


def calInternalSimilarity(recommendations: torch.Tensor, item_degree_dict: dict, 
                          interaction_mat: np.ndarray, k: int) -> float:
    """
    @description 计算多样性指标 - intra_similarity（内部相似性）
    @param {torch} recommendations 所有用户的TopK推荐
    @param {int} k 推荐列表大小
    @param {dict} item_degree_dict 物品的度
    @param {np.ndarray} interaction_mat 用户物品的交互矩阵
    @return {float} intra_similarity 内部相似性
    """

    # 用户数
    user_num, item_num = recommendations.shape

    # 总内部相似性
    total_I = 0.0

    # 遍历所有用户
    for uid in range(user_num):
        # 获取当前用户的推荐列表
        recommend_item = recommendations[uid].tolist()

        # 遍历推荐列表中的所有项目
        for iid_i in recommend_item:
            for iid_j in recommend_item:
                if iid_i == iid_j:
                    continue

                # 获取项目的度
                k_i = item_degree_dict.get(iid_i, 0)
                k_j = item_degree_dict.get(iid_j, 0)

                # 若度为 0，跳过该项目对
                if k_i == 0 or k_j == 0:
                    continue

                # 两项目度的乘积的平方根
                denom = np.sqrt(k_i * k_j)

                # 两物品的共同偏好数量，即同时交互物品i和物品j的用户数
                common_preference_count = np.dot(interaction_mat[:, iid_i], interaction_mat[:, iid_j])

                # 计算当前物品对的 Sørensen 距离
                s_ij = common_preference_count / denom
                total_I += s_ij


    I = total_I / (user_num *  k *(k - 1))

    return round(I, 5)

def getDiversityMetrics(recommendations: torch.Tensor, item_degree_dict: dict, 
                        interaction_mat: np.ndarray, k: int) -> tuple:
    """
    @description 计算多样性指标
    @param {torch} recommendations 所有用户的TopK推荐
    @param {int} k 推荐列表大小
    @param {dict} item_degree_dict 物品的度
    @param {np} interaction_mat 用户物品的交互矩阵
    @return {tuple} 元组包含
            - {float} H 海明距离
            - {float} I 内部相似性
    """

    # 计算海明距离
    H = calHammingDistance(recommendations, k)

    # 计算内部相似性
    I = calInternalSimilarity(recommendations, item_degree_dict, interaction_mat, k)

    return H, I
