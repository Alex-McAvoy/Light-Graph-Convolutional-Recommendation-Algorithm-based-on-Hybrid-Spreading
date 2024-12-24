#!/usr/bin/env python
# coding=utf-8
"""
@Description   movielens数据集数据处理
@Author        Alex_McAvoy
@Date          2024-09-26 21:42:19
"""

import numpy as np
import pandas as pd

from const import cfg
from utils.wrapper import calTimes
from utils.log import logger
from processing.handleData import handleRating
from processing.handleFeature import ageMap, yearMap, preprocessText, getWord2Vec


@calTimes(logger, "MovieLens数据集用户特征处理完成")
def movielensUserFeature(
    user_df: pd.DataFrame, occupation_df: pd.DataFrame
) -> pd.DataFrame:
    """
    @description movielens数据集用户特征提取
    @param {pd} user_df movielens数据集用户数据
    @param {pd} occupation_df movielens数据集职业信息数据
    @return {pd} 用户特征向量 ["user_id", "user_features"]
    """
    # 删除无用列
    del user_df["zip_code"]

    # 职业映射表
    occupation_map = {
        value: index
        for index, value in zip(occupation_df.index, occupation_df["occupation"])
    }

    # 年龄映射
    user_df["age"] = user_df["age"].apply(lambda age: ageMap(age))
    # 性别映射
    user_df["gender"] = user_df["gender"].map({"M": 1, "F": 0})
    # 职业映射
    user_df["occupation"] = user_df["occupation"].apply(
        lambda occupation: occupation_map[occupation]
    )

    # 年龄、职业转为one-hot编码
    user_features = pd.get_dummies(
        user_df.drop(columns=["user_id"]), columns=["age", "occupation"], dtype=int
    ).to_numpy()

    # 设置Dataframe结构
    user_features_df = user_df[["user_id"]].copy()
    user_features_df["user_features"] = [
        user_features[i].tolist() for i in range(user_features.shape[0])
    ]

    return user_features_df


@calTimes(logger, "MovieLens数据集项目特征处理完成")
def movielensItemFeature(
    item_df: pd.DataFrame,
    vector_size: int = cfg.PREPROCESSING["vector_size"]["title"],
) -> np.ndarray:
    """
    @description movielens数据集项目特征提取
    @param {pd} item_df movielens数据集项目数据
    @param {int} vector_size Word2Vec向量大小
    @return {np} 项目特征向量 ["item_id", "item_features"]
    """
    # 删除无用列
    del item_df["video_release_date"]
    del item_df["IMDb_URL"]

    # 电影名预处理
    item_df["movie_title"] = item_df["movie_title"].apply(preprocessText)
    # 清洗上映时间
    item_df["release_date"] = (
        item_df["release_date"].astype(str).apply(lambda row: row[-4:])
    )
    item_df["release_date"] = item_df["release_date"].apply(lambda year: yearMap(year))

    # 对上映时间进行one-hot
    item_features = pd.get_dummies(
        item_df.drop(columns=["movie_id", "movie_title"]),
        columns=["release_date"],
        dtype=int,
    ).to_numpy()
    # 对电影名进行Word2Vec
    title_encode = getWord2Vec(item_df["movie_title"].tolist(), vector_size)
    # 项目特征向量，拼接上映时间one-hot编码和电影名Word2Vec特征编码
    item_features = np.concatenate((item_features, title_encode), axis=1)

    # 设置Dataframe结构
    item_features_df = item_df[["movie_id"]].copy()
    item_features_df.rename(columns={"movie_id": "item_id"}, inplace=True)
    item_features_df["item_features"] = [
        item_features[i].tolist() for i in range(item_features.shape[0])
    ]

    

    return item_features_df


@calTimes(logger, "MovieLens数据集处理完成")
def prepareMovieLens(dataset_path_dict: dict, save_path: str) -> tuple:
    """
    @description 处理MovieLens数据集
    @param {dict} dataset_path_dict 数据集存储目录
    @param {str} save_path 处理数据存储目录
    @return {tuple} 元组包含
        - {pd} filtered_rating_df 清洗后的用户-评分数据
        - {pd} train_data_df 训练集
        - {pd} val_data_df 验证集
        - {pd} test_data_df 测试集
        - {pd} user_features_df 用户特征
        - {pd} item_features_df 项目特征
    """
    # 读取用户-项目评分信息
    rating_df = pd.read_csv(
        dataset_path_dict["rating"],
        sep="\t",
        header=None,
        names=["user", "item", "rating", "timestamp"],
    )
    rating_df["timestamp"] = pd.to_datetime(rating_df["timestamp"], unit="s")
    # 读取用户信息
    user_df = pd.read_csv(
        dataset_path_dict["users"],
        sep="|",
        header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )
    # 读取职业信息
    occupation_df = pd.read_csv(
        dataset_path_dict["occupation"], sep="\t", header=None, names=["occupation"]
    )
    # 读取项目信息
    item_df = pd.read_csv(
        dataset_path_dict["items"],
        sep="|",
        header=None,
        encoding="iso-8859-1",
        names=[
            "movie_id",
            "movie_title",
            "release_date",
            "video_release_date",
            "IMDb_URL",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children's",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ],
    )

    # 划分训练集与测试集
    rating_df, train_data_df, val_data_df, test_data_df, uid_mapping, iid_mapping = handleRating(rating_df, save_path)
    
    # 获取用户特征
    user_features_df = movielensUserFeature(user_df, occupation_df)
    # 获取训练集项目特征
    item_features_df = movielensItemFeature(item_df)
    # 转为内部id
    user_features_df.user_id = user_features_df.user_id.map(uid_mapping)
    item_features_df.item_id = item_features_df.item_id.map(iid_mapping)

    # 过滤到未找到的用户、项目特征
    user_features_df = user_features_df.dropna(subset=["user_id"])
    item_features_df = item_features_df.dropna(subset=["item_id"])

    # 存储数据
    user_features_df.to_csv(
        save_path + "user_features.csv", sep="\t", index=False, header=True
    )
    item_features_df.to_csv(
        save_path + "item_features.csv", sep="\t", index=False, header=True
    )

    return (
        rating_df,
        train_data_df,
        val_data_df,
        test_data_df,
        user_features_df,
        item_features_df,
    )
