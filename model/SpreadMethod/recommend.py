#!/usr/bin/env python
# coding=utf-8
"""
@Description   扩散方法推荐
@Author        Alex_McAvoy
@Date          2024-10-17 17:59:54
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from const import cfg
from model.SpreadMethod.model import getSpreadingGeneralMat, ProbS, HeatS, HybridS, getResource
from utils.trans import getUserItemsDictByDataframe, getInteractionMatrixByDataframe


def recommendForAllUser(F_new: np.ndarray, user_num:int, train_data_df: pd.DataFrame, 
                        val_data_df: pd.DataFrame, k: int) -> dict:
    """
    @description 获取所有用户的推荐字典
    @param {np} F_new 扩散后的资源矩阵，shape(user_num, item_num)
    @param {int} user_num 总用户数
    @param {int} item_num 总项目数
    @param {pd} train_data_df 训练数据
    @param {pd} val_data_df 验证数据
    @param {int} k 推荐列表大小
    @return {dict} 所有用户的推荐字典
    """
    # 获取训练集用户-项目交互字典
    train_user_item_dict = getUserItemsDictByDataframe(pd.concat([train_data_df, val_data_df]))

    all_user_recommend_dict = defaultdict(list)
    # 遍历所有用户
    for uid in range(user_num):
        # 获取该用户的资源值
        resource = F_new[uid]
        # 按照资源值，对项目id进行排序，索引为项目id
        sorted_items = np.argsort(resource)[::-1]

        # 获取该用户交互过的项目id
        interacted_items = train_user_item_dict.get(uid, [])
        # 过滤掉用户已交互过的项目
        filtered_items = [iid for iid in sorted_items if iid not in interacted_items]

        # 取前k个未交互过的项目作为推荐
        all_user_recommend_dict[uid] = filtered_items[:k]
        # movielens数据集清洗后项目较少，导致ProbS第一轮扩散计算项目出度时值极低，使得模型迅速欠拟合，各指标近乎以斜率1急剧下降，因此将转移矩阵转置后使用混合扩散从热扩散的反向近似
        if cfg.DATA_SET == "movielens" and cfg.MODEL["name"] == "ProbS":
            all_user_recommend_dict[uid] = sorted_items[:k]
  

    # 存储推荐结果
    np.save(cfg.RECOMMEND["save_path"] + "all_user_recommend_dict_" + cfg.MODEL["name"] + "_" + str(cfg.RECOMMEND["k"]) + ".npy", all_user_recommend_dict)

    return all_user_recommend_dict


def recommendSpreadMethod(user_num:int, item_num: int,
                          train_data_df: pd.DataFrame, val_data_df: pd.DataFrame,
                          method: str, lambda_val: float = 0) -> dict:
    """
    @description 扩散方法推荐
    @param {int} user_num 总用户数
    @param {int} item_num 总项目数
    @param {pd} train_data_df 训练数据
    @param {pd} val_data_df 验证数据
    @param {str} method 扩散方法
    @param {float} lambda_val HybridS 混合比例常数
    @return {dict} 所有用户的推荐字典
    """
    # 超参
    k = cfg.RECOMMEND["k"]                              # 推荐列表大小
    lambda_val = cfg.MODEL["HyperParameter"]["lambda"]  # 混合比例常数

    # 选择概率扩散、热扩散、混合扩散
    if method not in ["ProbS", "HeatS", "HybridS"]:
        raise ValueError(f"Invalid parameter: method={method}，必须为 ProbS | HeatS | HybridS")
    
    # 根据训练集和测试集，获取用户-项目交互矩阵
    A = getInteractionMatrixByDataframe(user_num, item_num, pd.concat([train_data_df, val_data_df]))

    # 计算通用扩散矩阵
    general_W = getSpreadingGeneralMat(A)

    # 概率扩散
    if method == "ProbS":
        # movielens数据集清洗后项目较少，导致ProbS第一轮扩散计算项目出度时值极低，使得模型迅速欠拟合，各指标近乎以斜率1急剧下降，因此将转移矩阵转置后使用混合扩散从热扩散的反向近似
        if cfg.DATA_SET == "movielens":
            lambda_val = 0.01
            general_W = general_W.T
        # 获取转移矩阵
        W =  HybridS(A, general_W, lambda_val)
        # 获取扩散后的资源值
        F_new = getResource(A, W)
    # 热扩散
    elif method == "HeatS":
        # douban数据集清洗后用户-项目交互数据较少，导致HeatS第二轮计算项目入度时值极低，使得模型迅速过拟合，各指标近乎以斜率1急剧上涨，因此将转移矩阵转置后使用混合扩散从概率扩散的反向近似
        if cfg.DATA_SET == "douban":
            lambda_val = 0.99
            general_W = general_W.T
        # 获取转移矩阵
        W =  HybridS(A, general_W, lambda_val)
        # 获取扩散后的资源值
        F_new = getResource(A, W)
    # 混合扩散
    elif method =="HybridS":
        # 获取转移矩阵
        W = HybridS(A, general_W, lambda_val)
        # 获取扩散后的资源值
        F_new = getResource(A, W)

    all_user_recommend_dict = recommendForAllUser(F_new, user_num, train_data_df, val_data_df, k)

    return all_user_recommend_dict

