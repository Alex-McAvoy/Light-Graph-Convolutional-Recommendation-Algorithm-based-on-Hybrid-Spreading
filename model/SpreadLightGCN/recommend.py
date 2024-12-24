#!/usr/bin/env python
# coding=utf-8
"""
@Description   扩散+LightGCN方法推荐
@Author        Alex_McAvoy
@Date          2024-10-17 17:59:54
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import torch

from const import cfg
from model.SpreadLightGCN.model import getResourceMat
from utils.trans import getUserItemsDictByDataframe

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

    # 存储推荐结果
    np.save(cfg.RECOMMEND["save_path"] + "all_user_recommend_dict_" + cfg.MODEL["name"] + "_" + str(cfg.RECOMMEND["k"]) + ".npy", all_user_recommend_dict)

    return all_user_recommend_dict


def recommendSpreadLightGCN(user_num:int, item_num: int, rating_df: pd.DataFrame,
                            train_data_df: pd.DataFrame, val_data_df: pd.DataFrame, 
                            test_data_df: pd.DataFrame) -> dict:
    """
    @description 扩散+LightGCN推荐
    @param {int} user_num 总用户数
    @param {int} item_num 总项目数
    @param {pd} rating_df 过滤后的评分数据
    @param {pd} train_data_df 训练数据
    @param {pd} val_data_df 验证数据
    @param {pd} val_data_df 测试数据
    @return {dict} 所有用户的推荐字典
    """
    # 超参
    k = cfg.RECOMMEND["k"] # 推荐列表大小
    # 计算资源转移矩阵
    F_new = getResourceMat(user_num, item_num, rating_df, train_data_df, val_data_df, test_data_df)
    # 获取推荐字典
    all_user_recommend_dict = recommendForAllUser(F_new, user_num, train_data_df, val_data_df, k)

    return all_user_recommend_dict
