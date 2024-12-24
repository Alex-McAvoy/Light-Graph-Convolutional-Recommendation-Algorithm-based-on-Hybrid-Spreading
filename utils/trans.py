#!/usr/bin/env python
# coding=utf-8
"""
@Description   转换工具
@Author        Alex_McAvoy
@Date          2024-10-17 21:44:09
"""
import numpy as np
import pandas as pd
from collections import defaultdict
import torch

def getInteractionMatrixByDataframe(user_num:int, item_num: int, data_df: pd.DataFrame) -> np.ndarray:
    """
    @description 根据用户-项目评分Dataframe，构建用户-项目交互矩阵
    @param {int} user_num 用户数
    @param {int} item_num 项目数
    @param {pd} data_df 用户-项目评分
    @return {np} 用户-项目交互矩阵
    """
    # 初始化用户-项目交互矩阵
    A = np.zeros((user_num, item_num))
    # 根据用户-项目评分填充交互矩阵
    for idx, row in data_df.iterrows():
        uid = row["user_id"]
        iid = row["item_id"]
        A[uid, iid] = 1

    return A

def getInteractionMatrixByEdgeIndex(user_num:int, item_num: int, edge_index: torch.Tensor) -> np.ndarray:
    """
    @description 根据边索引，构建用户-项目交互矩阵
    @param {int} user_num 用户数
    @param {int} item_num 项目数
    @param {pd} data_df 用户-项目评分
    @return {np} 用户-项目交互矩阵
    """
    # 初始化用户-项目交互矩阵
    A = np.zeros((user_num, item_num))
    # 获取用户和项目id列表
    uids = edge_index[0].tolist()
    iids = edge_index[1].tolist()

    # 填充交互矩阵
    for uid, iid in zip(uids, iids):
        A[int(uid), int(iid)] = 1
        
    return A

def getUserItemsDictByDataframe(data_df: pd.DataFrame) -> dict:
    """
    @description 根据用户-项目评分Dataframe，构建用户-项目交互字典
    @param {pd} data_df 用户-项目评分
    @return {dict} 用户-项目字典
    """
    # 获取用户-项目字典
    user_items_dict = defaultdict(list)
    for idx, row in data_df.iterrows():
        uid = row["user_id"]
        iid = row["item_id"]
        user_items_dict[uid].append(iid)
    return user_items_dict

def getUserItemsDictByEdgeIndex(edge_index: torch.Tensor) -> dict:
    """
    @description 获取用户-项目字典
    @param {torch} edge_index 边索引
    @return {dict} 用户-项目字典
    """
    user_items_dict = {}    
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        
        if user not in user_items_dict:
            user_items_dict[user] = []
        user_items_dict[user].append(item)
        
    return user_items_dict

def recommendDictToTensor(recommend_dict: dict) -> torch.Tensor:
    """
    @description 将所有用户的推荐字典转为tensor格式
    @param {dict} target_dict 所有用户的推荐字典
    @return {tensor} 所有用户的推荐tensor，recommendations[i]代表用户id为i的推荐列表
    """
    # 将推荐字典按照用户id排序，并提取对应的推荐物品列表
    sorted_items = [recommend_dict[uid] for uid in sorted(recommend_dict.keys())]
    # 转为tensor
    recommendations = torch.tensor(np.array(sorted_items))
    return recommendations

def getItemDegreeByUserPosItemDict(*user_pos_items_dict_list: dict) -> dict:
    """
    @deprecated 通过用户-物品字典计算物品的度
    @param {list} *user_pos_items_dict_list user_pos_items_dict列表
        - {dict} user_pos_items_dict 用户的正样本物品字典，user_pos_items_dict[i]代表用户i的正样本物品列表
    @return {dict} 物品的度
    """
    # 物品的度
    item_degree_dict = {}

    # 遍历传入的每一个用户的正样本物品字典
    for user_pos_items_dict in user_pos_items_dict_list:
        # 遍历每个用户的物品
        for uid, items in user_pos_items_dict.items():
            # 遍历每个物品
            for item in items:
                # 计算物品的度
                if item in item_degree_dict:
                    item_degree_dict[item] += 1
                else:
                    item_degree_dict[item] = 1

    return item_degree_dict