#!/usr/bin/env python
# coding=utf-8
"""
@Description   准确性评估指标
@Author        Alex_McAvoy
@Date          2024-10-11 20:33:40
"""

import pandas as pd

def calculateTpFpFn(test_data_df: pd.DataFrame, all_user_recommend_dict: dict) -> tuple:
    """
    @description 计算TP、FP、FN，以用于计算Precision、Recall、F1-score
    @param {pd} test_data_df 测试集数据
    @param {dict} all_user_recommend_dict 训练集所有用户的推荐列表
    @return {tuple} 元组包含
            - {int} total_tp 推荐列表中正确推荐的物品数量
            - {int} total_fp 推荐列表中错误推荐的物品数量
            - {int} total_fn 未被推荐但在测试集中实际交互的物品数量
    """
    # TP：推荐列表中正确推荐的物品数量
    total_tp = 0
    # FP：推荐列表中错误推荐的物品数量
    total_fp = 0
    # FN：未被推荐但在测试集中实际交互的物品数量
    total_fn = 0

    # 遍历所有用户的推荐列表
    for uid, recommend_items in all_user_recommend_dict.items():
        # 获取当前用户在测试集中的真实交互
        true_items = set(test_data_df[test_data_df["user_id"] == uid]["item_id"])

        # 计算TP
        tp = len(set(recommend_items) & true_items)
        total_tp += tp

        # 计算FP
        fp = len(set(recommend_items)) - tp
        total_fp += fp

        # 计算FN
        fn = len(true_items) - tp
        total_fn += fn

    return total_tp, total_fp, total_fn

def calPrecision(tp: int, fp: int) -> float:
    """
    @description 计算精确率
    @param {int} tp 推荐列表中正确推荐的物品数量
    @param {int} fp 推荐列表中错误推荐的物品数量
    @return {float} 精确率
    """
    if tp + fp > 0:
        precision = round(tp / (tp + fp), 5)
    else:
        precision = 0.0
    return precision

def calRecall(tp: int, fn: int) -> float:
    """
    @description 计算召回率
    @param {int} tp 推荐列表中正确推荐的物品数量
    @param {int} fn 未被推荐但在测试集中实际交互的物品数量
    @return {float} 召回率
    """
    if tp + fn > 0:
        recall = round(tp / (tp + fn), 5)
    else:
        recall = 0.0
    return recall

def calF1score(precision: float, recall: float) -> float:
    """
    @description 计算F1得分
    @param {float} precision 精确率
    @param {float} recall 召回率
    @return {float} F1得分
    """
    if precision + recall > 0:
        f1 = round(2 * (precision * recall) / (precision + recall), 5)
    else:
        f1 = 0.0
    return f1

def calAveragedRankingScore(test_data_df: pd.DataFrame, all_user_recommend_dict: dict) -> float:
    """
    @description 计算平均排名得分
    @param {pd} test_data_df 测试集数据
    @param {dict} all_user_recommend_dict 训练集所有用户的推荐列表
    @return {float} 平均排名得分
    """
    # 排名得分总分
    total_r = 0
    # 遍历测试集中的每个用户-物品关系对
    for uid, iid in zip(test_data_df['user_id'], test_data_df['item_id']):
        # 获取当前用户的推荐列表
        recommend_items = all_user_recommend_dict[uid]
        # 获取当前用户的推荐列表长度
        L_i = len(recommend_items)

        # 若当前物品不在推荐列表中，不计算该用户-物品关系对
        if iid not in recommend_items:
            continue

        # 计算当前物品在推荐列表中的排名
        rank = recommend_items.index(iid) + 1
        # 计算排名得分
        r = rank / L_i
        total_r += r
    
    # 平均排名得分
    r = round(total_r / len(test_data_df), 5)

    return r

def getAccurateMetrics(test_data_df: pd.DataFrame, all_user_recommend_dict: dict) -> tuple:
    """
    @description 获取准确性相关指标
    @param {pd} test_data_df 测试集数据
    @param {dict} all_user_recommend_dict 训练集所有用户的推荐列表
    @return {tuple} 元组包含
            - {float} precision 精确率
            - {float} recall 召回率
            - {float} f1 F1得分
            - {float} r 平均排名得分
    """
    # TP、FP、FN
    tp, fp, fn = calculateTpFpFn(test_data_df, all_user_recommend_dict)
    # 精确率
    precision = calPrecision(tp, fp)
    # 召回率
    recall = calRecall(tp, fn)
    # F1得分
    f1 = calF1score(precision, recall)
    # 平均排名得分
    r = calAveragedRankingScore(test_data_df, all_user_recommend_dict)

    return precision, recall, f1, r