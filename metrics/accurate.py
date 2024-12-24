#!/usr/bin/env python
# coding=utf-8
"""
@Description   计算准确性相关指标
@Author        Alex_McAvoy
@Date          2024-10-16 16:48:55
"""
import numpy as np
import torch

def calPrecisionAndRecall(user_pos_items_dict: dict, recommendations: torch.Tensor, k: int) -> tuple:
    """
    @description 计算精确率与召回率
    @param {dict} user_pos_items_dict 用户的正样本物品字典，user_pos_items_dict[i]代表用户i的正样本物品列表
    @param {torch} recommendations 所有用户的TopK推荐
    @param {int} k 推荐列表大小
    @return {tuple} 元组包含
            - {float} precision 精确率
            - {float} recall 召回率
    """
    # 用户的推荐交互标记，recommend_interaction_list[i]为用户i推荐的前k个物品中为真实交互过的物品标记
    recommend_interaction_list = []
    # 每个用户的真实正样本数量
    user_num_liked_list = []
    # 遍历所有用户与真实交互过的物品集
    for uid, items in user_pos_items_dict.items():
        # 检查推荐物品是否在交互列表中，若在标记为True，否则标记为False
        label = list(map(lambda item: item in items, recommendations[uid]))
        recommend_interaction_list.append(label)
        # 计算每个用户的真实交互样本数量
        user_true_relevant_item_num = len(items)
        user_num_liked_list.append(user_true_relevant_item_num)
    recommend_interaction_list = torch.Tensor(np.array(recommend_interaction_list).astype('float'))
    user_num_liked_list = torch.Tensor(user_num_liked_list)

    # 计算每个用户预测正确的物品数量
    num_correct_pred = torch.sum(recommend_interaction_list, dim=-1)  
    
    # 计算精确率
    precision = torch.mean(num_correct_pred) / k
    # 计算召回率
    recall = torch.mean(num_correct_pred / user_num_liked_list)

    

    return round(precision.item(), 5), round(recall.item(), 5)

def calF1Score(precision: float, recall: float) -> float:
    """
    @description 计算F1得分
    @param {float} precision 精确率
    @param {float} recall 召回率
    @return {float} F1得分
    """
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 5)

def calNDCG(user_pos_items_dict: dict, recommendations: torch.Tensor, k: int) -> float:
    """
    @description 计算归一化折损累积增益
    @param {dict} user_pos_items_dict 用户的正样本物品字典，user_pos_items_dict[i]代表用户i的正样本物品列表
    @param {torch} recommendations 所有用户的TopK推荐
    @param {int} k 推荐列表大小
    @return {float} 归一化折损累积增益
    """
    # 用户的推荐交互标记，recommend_interaction_list[i]为用户i推荐的前k个物品中为真实交互过的物品标记
    recommend_interaction_list = []
    # 遍历所有用户与真实交互过的物品集
    for uid, items in user_pos_items_dict.items():
        # 检查推荐物品是否在交互列表中，若在标记为True，否则标记为False
        label = list(map(lambda item: item in items, recommendations[uid]))
        recommend_interaction_list.append(label)
    recommend_interaction_list = torch.Tensor(np.array(recommend_interaction_list).astype('float'))

    # 遍历每个用户的推荐交互列表
    tmp_matrix = torch.zeros((len(recommend_interaction_list), k))
    for index, items in enumerate(recommend_interaction_list):
        # 获取当前用户推荐的前k个物品数量
        length = min(len(items), k)
        # 对前length个位置赋1，表示这些位置的推荐物品被视为最高相关
        tmp_matrix[index, :length] = 1
    
    # 理想情况下的DCG，推荐的都为相关物品
    max_r = tmp_matrix
    # 计算IDCG
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2)), axis=1)

    # 计算实际的DCG
    dcg = recommend_interaction_list * (1. / torch.log2(torch.arange(2, k + 2)))
    # 按列求和，得每个用户的DCG
    dcg = torch.sum(dcg, axis=1)


    # 计算NDCG
    idcg[idcg == 0.] = 1.  # 避免分母为零
    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0. # 防止无效的计算结果

    # 计算所有用户的平均NDCG
    ndcg = torch.mean(ndcg)

    return round(ndcg.item(), 5)

def getAccurateMetrics(user_pos_items_dict: dict, recommendations: torch.Tensor, k: int) -> tuple:
    """
    @description 计算准确率指标
    @param {dict} user_pos_items_dict 用户的正样本物品字典，user_pos_items_dict[i]代表用户i的正样本物品列表
    @param {torch} recommendations 所有用户的TopK推荐
    @param {int} k 推荐列表大小
    @return {tuple} 元组包含
            - {float} precision 精确率
            - {float} recall 召回率
            - {float} f1 F1得分
            - {float} ndcg 归一化折损累积增益
    """

    # 计算精确率，召回率
    precision, recall = calPrecisionAndRecall(user_pos_items_dict, recommendations, k)

    # 计算F1得分
    f1 = calF1Score(precision, recall)

    # 计算归一化折损累积增益
    ndcg = calNDCG(user_pos_items_dict, recommendations, k)

    return precision, recall, f1, ndcg

