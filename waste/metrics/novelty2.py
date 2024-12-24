#!/usr/bin/env python
# coding=utf-8
"""
@Description   新颖性指标评估
@Author        Alex_McAvoy
@Date          2024-10-17 22:09:56
"""
import torch

from utils.log import logger
from utils.wrapper import calTimes


def calAverageDegree(recommendations: torch.Tensor, item_degree_dict: dict, k: int) -> float:
    """
    @description 计算多样性指标 - hamming_distance(海明距离)
    @param {dict} item_degree_dict 物品的度
    @param {torch} recommendations 所有用户的TopK推荐
    @param {int} k 推荐列表大小
    @return {float} hamming_distance 海明距离
    """

    # 用户数
    user_num, _ = recommendations.shape

    # 总的度
    total_degree = 0.0

    for user in range(user_num):

        # 用户user的推荐列表
        recommend_items = recommendations[user].tolist()

        # 遍历推荐列表
        for item in recommend_items:
            total_degree += item_degree_dict.get(item, 0)

    average_degree = total_degree / (user_num * k)

    return round(average_degree, 5)


def getNoveltyMetrics(recommendations: torch.Tensor, item_degree_dict: dict, k: int) -> float:
    """
    @description 计算新颖性指标
    @param {torch} recommendations 所有用户的TopK推荐
    @param {dict} item_degree_dict 物品的度
    @param {int} k 推荐列表大小
    @return {float} average_degree 平均度
    """

    # 平均度
    average_degree = calAverageDegree(recommendations, item_degree_dict, k)

    return average_degree
