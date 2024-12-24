#!/usr/bin/env python
# coding=utf-8
"""
@Description   新颖性指标
@Author        Alex_McAvoy
@Date          2024-10-12 20:43:16
"""

from waste.processing.dataset import Dataset
    
def calAverageDegree(train_dataset: Dataset, all_user_recommend_dict: dict, L: int) -> float:
    """
    @description 计算平均度
    @param {Dataset} train_dataset 内部数据集
    @param {dict} all_user_recommend_dict 训练集所有用户的推荐列表
    @param {int} L 用户推荐列表大小
    @return {float} 平均度
    """
    # 物品的度
    item_degrees = train_dataset.data_df["item_id"].value_counts().to_dict()
    # 总度
    total_degree = 0

    # 遍历所有用户的推荐列表
    for uid, recommend_items in all_user_recommend_dict.items():
        # 遍历推荐列表
        for iid in recommend_items:
            # 转为内部id
            idx = train_dataset.iid2idx_map.get(iid, -1)
            # 获取物品的度
            degree = item_degrees.get(idx, 0)
            total_degree += degree

    average_degree = round(total_degree / (L * len(all_user_recommend_dict)), 5)

    return average_degree


def getNoveltyMetrics(train_dataset: Dataset, all_user_recommend_dict: dict, k: int) -> float:
    """
    @description 获取新颖性相关指标
    @param {Dataset} train_dataset 内部数据集
    @param {dict} all_user_recommend_dict 训练集所有用户的推荐列表
    @param {int} k 用户推荐列表大小
    @return {float} 平均度
    """
    average_degree = calAverageDegree(train_dataset, all_user_recommend_dict, k)

    return average_degree