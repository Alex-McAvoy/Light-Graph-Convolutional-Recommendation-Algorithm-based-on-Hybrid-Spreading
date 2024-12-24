#!/usr/bin/env python
# coding=utf-8
"""
@Description   内部数据集
@Author        Alex_McAvoy
@Date          2024-09-29 20:22:22
"""
import numpy as np
import pandas as pd
import ast

from utils.log import logger
from utils.wrapper import calTimes

@calTimes(logger, "内部数据集构建完成")
class Dataset(object):
    def __init__(
        self,
        users_id_np: np.ndarray,
        items_id_np: np.ndarray,
        data_df: pd.DataFrame,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> None:
        """
        @description 内部数据集
        @param {*} self 类实例对象
        @param {np} users 用户id列表（外部id）
        @param {np} items 项目id列表（外部id）
        @param {pd} data_df 数据集
        @param {pd} user_features_df 用户特征列表（外部id）
        @param {pd} item_features_df 项目特征列表（外部id）
        """
        # 用户外部id列表
        self._uid_np = users_id_np
        # 项目外部id列表
        self._iid_np = items_id_np

        self.users_num = users_id_np.shape[0]
        self.items_num = items_id_np.shape[0]

        # 构建用户索引
        self._uid2idx_map, self._idx2uid_map = self._buildUid2Idx()
        # 构建项目索引
        self._iid2idx_map, self._idx2iid_map = self._buildIid2Idx()

        # 用户内部id列表
        self._uIdx_np = self._getUsersIndex()
        # 项目内部id列表
        self._iIdx_np = self._getItemsIndex()

        # 数据集中用户id、项目id替换为内部id
        self._data_df = self._getData(data_df)

        # 用户特征、项目特征，users_features_np[i]为内部用户id为i的用户特征向量，items_features_np[i]为内部项目id为i的用户特征向量
        self._users_features_np, self._items_features_np = self._getFeatures(user_features_df, item_features_df)


    def _buildUid2Idx(self) -> tuple:
        """
        @description 构建用户id索引
        @param {*} self 类实例对象
        @return {tuple} 元组包含
            - {dict} uid2idx_map 用户id-内部id字典
            - {dict} idx2uid_map 内部id-用户id字典
        """
        # 用户id到index的映射关系，index从0开始编码
        uid2idx_map = dict()
        # index到用户id的映射关系，index从0开始编码
        idx2uid_map = dict()

        index = 0
        for uid in self._uid_np:
            uid2idx_map[uid] = index
            idx2uid_map[index] = uid
            index = index + 1

        return uid2idx_map, idx2uid_map

    def _buildIid2Idx(self) -> tuple:
        """
        @description 构建项目id索引
        @param {*} self 类实例对象
        @return {tuple} 元组包含
            - {dict} iid2idx_map 项目id-内部id字典
            - {dict} idx2iid_map 内部id-项目id字典
        """
        # 项目id到index的映射关系，index从0开始编码
        iid2idx_map = dict()
        # index到项目id的映射关系，index从0开始编码
        idx2iid_map = dict()

        index = 0
        for iid in self._iid_np:
            iid2idx_map[iid] = index
            idx2iid_map[index] = iid
            index = index + 1

        return iid2idx_map, idx2iid_map

    def _getUsersIndex(self) -> np.ndarray:
        """
        @description 构建用户内部id列表
        @param {*} self 类实例化对象
        @return {np} 用户内部id列表
        """
        return np.array([self._uid2idx_map[item] for item in self._uid_np])

    def _getItemsIndex(self) -> np.ndarray:
        """
        @description 构建项目内部id列表
        @param {*} self 类实例化对象
        @return {np} 项目内部id列表
        """
        return np.array([self._iid2idx_map[item] for item in self._iid_np])

    def _getData(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        @description 将数据集中的用户id、项目id替换为内部id
        @param {*} self 类实例化对象
        @param {pd} data_df 替换为内部id
        @return {pd} 用户id、项目id为内部id的数据集
        """
        data_df["user_id"]= data_df["user_id"].map(self._uid2idx_map)
        data_df["item_id"] = data_df["item_id"].map(self._iid2idx_map)
        return data_df

    def _getFeatures(self, user_features_df: pd.DataFrame, item_features_df: pd.DataFrame) -> tuple:
        """
        @description 
        @param {*} self 类实例化对象
        @param {*} user_features_df 用户特征，user_id为外部id
        @param {*} item_features_df 项目特征，item_id为外部id
        @return {tuple} 元组包含
                - {np} users_features_np 用户特征矩阵，users_features_np[i]为内部用户id为i的用户特征向量
                - {np} items_features_np 项目特征矩阵，items_features_np[i]为内部项目id为i的用户特征向量
        """

        # 训练集中的用户特征向量
        filter_user_features_df = user_features_df[user_features_df["user_id"].isin(self._uid_np)]           # 根据外部id只保留训练集中的数据
        filter_user_features_df.loc[:,"user_id"] = filter_user_features_df["user_id"].map(self._uid2idx_map) # 转为内部id
        filter_user_features_df_sorted = filter_user_features_df.sort_values(by="user_id")                   # 根据内部用户id排序
        users_features_np = filter_user_features_df_sorted["user_features"].apply(
            lambda row: ast.literal_eval(row) if not isinstance(row, list) else row
        ).tolist()                                                                                           # 转为用户特征列表
        users_features_np = np.array(users_features_np)                                                      # 转为numpy

        # 训练集中的项目特征向量
        filter_item_features_df = item_features_df[item_features_df["item_id"].isin(self._iid_np)]           # 根据外部id只保留训练集中的数据
        filter_item_features_df.loc[:,"item_id"] = filter_item_features_df["item_id"].map(self._iid2idx_map) # 转为内部id
        filter_item_features_df_sorted = filter_item_features_df.sort_values(by="item_id")                   # 根据内部项目id排序
        items_features_np = filter_item_features_df_sorted["item_features"].apply(
            lambda row: ast.literal_eval(row) if not isinstance(row, list) else row
        ).tolist()                                                                                           # 转为项目特征列表
        items_features_np = np.array(items_features_np)                                                      # 转为numpy

        # 扩展用户特征的列维度与物品的维度相同
        if items_features_np.shape[1] > users_features_np.shape[1]:
            zero_np = np.zeros((users_features_np.shape[0], items_features_np.shape[1] - users_features_np.shape[1]))
            users_features_np = np.c_[users_features_np, zero_np]

        return users_features_np, items_features_np

    @property
    def uid_np(self):
        return self._uid_np

    @property
    def iid_np(self):
        return self._iid_np

    @property
    def uIdx_np(self):
        return self._uIdx_np

    @property
    def iIdx_np(self):
        return self._iIdx_np

    @property
    def uid2idx_map(self):
        return self._uid2idx_map

    @property
    def idx2uid_map(self):
        return self._idx2uid_map

    @property
    def iid2idx_map(self):
        return self._iid2idx_map

    @property
    def idx2iid_map(self):
        return self._idx2iid_map
        
    @property
    def data_df(self):
        return self._data_df

    @property
    def users_features_np(self):
        return self._users_features_np
    
    @property
    def items_features_np(self):
        return self._items_features_np