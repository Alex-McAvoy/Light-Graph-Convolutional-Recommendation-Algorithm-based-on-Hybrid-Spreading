#!/usr/bin/env python
# coding=utf-8
"""
@Description   多样性评估指标
@Author        Alex_McAvoy
@Date          2024-10-11 22:07:33
"""

import numpy as np
from waste.processing.dataset import Dataset

def calHammingDistance(all_user_recommend_dict: dict, L: int) -> float:
    """
    @description 计算海明距离
    @param {dict} all_user_recommend_dict 训练集所有用户的推荐列表
    @param {int} L 用户推荐列表大小
    @return {float} 海明距离
    """

    # 用户数
    m = len(all_user_recommend_dict)
    # 用户id列表
    users = list(all_user_recommend_dict.keys())

    # 总海明距离
    total_H = 0.0

    # 枚举用户对
    for i in range(m):
        for j in range(i+1, m):
            # 用户i的推荐列表
            uid_i = users[i]
            recommend_items_i = set(all_user_recommend_dict[uid_i])

            # 用户j的推荐列表
            uid_j = users[j]
            recommend_items_j = set(all_user_recommend_dict[uid_j])

            # 计算用户i、用户j的推荐项目重叠数
            Q = len(recommend_items_i & recommend_items_j)

            # 计算用户i、用户j的海明距离
            hamming = 1 - (Q/L)
            total_H += hamming

    # 平均海明距离
    H = round(total_H / (m * (m - 1)), 5)
    
    return H

def calInternalSimilarity(train_dataset: Dataset, all_user_recommend_dict: dict, interaction_mat_np: np.ndarray, L: int) -> float:
    """
    @description 计算内部相似性
    @param {Dataset} train_dataset 内部数据集
    @param {dict} all_user_recommend_dict 训练集所有用户的推荐列表
    @param {np} interaction_mat_np 训练集的用户-物品交互矩阵
    @param {int} L 用户推荐列表大小
    @return {float} 内部相似性
    """
    # 总内部相似性
    total_s = 0.0
    # 物品的度
    item_degrees = np.sum(interaction_mat_np, axis=0)

    # 遍历所有用户的推荐列表
    for uid, recommend_items in all_user_recommend_dict.items():
        # 推荐列表中物品个数不足两个，跳过
        if len(recommend_items) < 2:
            continue
        
        # 计算用户推荐列表中任意两物品的 Sørensen 距离
        for i in range(len(recommend_items)):
            for j in range(len(recommend_items)):

                if i == j:
                    continue
                
                # 物品i的内部id
                iid_i = recommend_items[i]
                idx_i = train_dataset.iid2idx_map.get(iid_i, -1)
                # 物品j的内部id
                iid_j = recommend_items[j]
                idx_j = train_dataset.iid2idx_map.get(iid_j, -1)

                # 若推荐列表中的物品i和物品j，不在，跳过该物品对
                if idx_i == -1 or idx_j == -1:
                    continue

                # 物品的度
                k_i = item_degrees[idx_i]
                k_j = item_degrees[idx_j]

                # 若物品度为0，跳过该物品对
                if k_i == 0 or k_j == 0:
                    continue
            
                # 两物品度的乘积的平方根
                denom = np.sqrt(k_i * k_j)

                # 两物品的共同偏好数量，即同时交互物品i和物品j的用户数
                common_preference_count = np.dot(interaction_mat_np[:, idx_i], interaction_mat_np[:, idx_j])

                # 计算当前物品对的 Sørensen 距离
                s_ij = common_preference_count / denom
                total_s += s_ij

    # 平均内部相似性
    I = round(total_s / (len(all_user_recommend_dict) * L * (L - 1)), 5)

    return I

def getDiversityMetrics(train_dataset: Dataset, all_user_recommend_dict: dict, interaction_mat_np: np.ndarray, k: int) -> tuple:
    """
    @description 获取准确性相关指标
    @param {Dataset} train_dataset 内部数据集
    @param {dict} all_user_recommend_dict 训练集所有用户的推荐列表
    @param {np} interaction_mat_np 训练集的用户-物品交互矩阵
    @param {int} k 用户推荐列表大小
    @return {tuple} 元组包含
            - {float} H 海明距离
            - {float} L 内部相似性
    """
    # 海明距离
    H = calHammingDistance(all_user_recommend_dict, k)
    # 内部相似性
    I = calInternalSimilarity(train_dataset, all_user_recommend_dict, interaction_mat_np, k)

    return H, I