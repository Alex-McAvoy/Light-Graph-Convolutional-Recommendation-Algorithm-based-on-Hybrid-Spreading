#!/usr/bin/env python
# coding=utf-8
"""
@Description   douban数据集处理
@Author        Alex_McAvoy
@Date          2024-09-26 22:03:34
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from const import cfg
from utils.wrapper import calTimes
from utils.log import logger
from processing.handleData import handleRating
from processing.handleFeature import (
    yearMap,
    genreCleanMap,
    preprocessText,
    durationMap,
    languageMap,
    regionMap,
    getWord2Vec,
)


@calTimes(logger, "Douban数据集用户特征处理完成")
def doubanUserFeature(
    user_df: pd.DataFrame,
    vector_size: int = cfg.PREPROCESSING["vector_size"]["title"],
) -> pd.DataFrame:
    """
    @description movielens数据集用户特征提取
    @param {pd} user_df movielens数据集用户数据
    @param {str} save_path 存储路径
    @param {int} vector_size Word2Vec向量大小
    @return {pd} 用户特征向量 ["user_id"，"user_features"]
    """

    # 昵称预处理
    user_df["USER_NICKNAME"] = user_df["USER_NICKNAME"].apply(preprocessText)

    # 对昵称进行Word2Vec
    nickname_encode = getWord2Vec(user_df["USER_NICKNAME"].tolist(), vector_size)

    # 设置Dataframe结构
    user_features_df = user_df[["USER_MD5"]].copy()
    user_features_df.rename(columns={"USER_MD5": "user_id"}, inplace=True)
    user_features_df["user_features"] = [
        nickname_encode[i].tolist() for i in range(nickname_encode.shape[0])
    ]

    

    return user_features_df


@calTimes(logger, "Douban数据集项目特征处理完成")
def doubanItemFeature(
    item_df: pd.DataFrame,
    vector_size_title: int = cfg.PREPROCESSING["vector_size"]["title"],
    vector_size_content: int = cfg.PREPROCESSING["vector_size"]["content"],
) -> np.ndarray:
    """
    @description douban数据集项目特征提取
    @param {pd} item_df douban数据集项目数据
    @param {str} save_path 存储路径
    @param {int} vector_size Word2Vec向量大小
    @return {np} 项目特征向量 ["item_id", "item_features"]
    """

    # 删除无用列
    del item_df["OFFICIAL_SITE"]
    del item_df["DOUBAN_SCORE"]
    del item_df["DIRECTORS"]
    del item_df["DOUBAN_VOTES"]
    del item_df["ALIAS"]
    del item_df["ACTORS"]
    del item_df["COVER"]
    del item_df["IMDB_ID"]
    del item_df["ACTOR_IDS"]
    del item_df["DIRECTOR_IDS"]
    del item_df["RELEASE_DATE"]
    del item_df["TAGS"]
    del item_df["SLUG"]

    # 清洗类别
    item_df["GENRES"] = (
        item_df["GENRES"].fillna("").str.split(r"[ /]").apply(lambda x: x if x else [])
    )
    item_df["GENRES"] = item_df["GENRES"].apply(genreCleanMap)
    # 清洗语言
    item_df["LANGUAGES"] = (
        item_df["LANGUAGES"]
        .fillna("")
        .str.replace(" ", "")
        .str.split(r"[/ |]")
        .apply(lambda x: x if x else [])
    )
    item_df["LANGUAGES"] = item_df["LANGUAGES"].apply(languageMap)
    # 清洗发行地区
    item_df["REGIONS"] = (
        item_df["REGIONS"].fillna("").str.split(r"[/]").apply(lambda x: x if x else [])
    )
    item_df["REGIONS"] = item_df["REGIONS"].apply(regionMap)
    # 计算电影时长均值并补全
    mean_value = item_df["MINS"].replace(0.0, pd.NA).mean()
    item_df["MINS"].replace(0.0, mean_value, inplace=True)
    item_df["MINS"] = item_df["MINS"].apply(lambda mins: durationMap(mins))
    # 发行年份映射
    item_df["YEAR"].fillna(0, inplace=True)
    item_df["YEAR"] = item_df["YEAR"].apply(lambda year: yearMap(year))
    # 预处理电影名
    item_df["NAME"] = item_df["NAME"].apply(preprocessText)
    # 预处理电影介绍
    item_df["STORYLINE"] = item_df["STORYLINE"].apply(preprocessText)

    # 多标签编码器
    mlb = MultiLabelBinarizer()
    # 类别多标签编码
    genre_encoded = mlb.fit_transform(item_df["GENRES"])
    # 语言多标签编码
    language_encoded = mlb.fit_transform(item_df["LANGUAGES"])
    # 发行地区多标签编码
    regions_encoded = mlb.fit_transform(item_df["REGIONS"])
    # 电影时长one-hot编码
    duration_encode = pd.get_dummies(item_df["MINS"], dtype=int)
    # 发行时间one-hot编码
    year_encode = pd.get_dummies(item_df["YEAR"], dtype=int)
    # 电影名Word2Vec编码
    name_encode = getWord2Vec(item_df["NAME"].tolist(), vector_size_title)
    # 电影介绍Word2Vec编码
    storyline_encode = getWord2Vec(item_df["STORYLINE"].tolist(), vector_size_content)

    # 项目特征向量，拼接 [电影名，类别，语言，时长，发行地区，介绍，发行时间]
    item_features = np.concatenate(
        [
            name_encode,
            genre_encoded,
            language_encoded,
            duration_encode,
            storyline_encode,
            regions_encoded,
            year_encode,
        ],
        axis=1,
    )

    # 设置Dataframe结构
    item_features_df = item_df[["MOVIE_ID"]].copy()
    item_features_df.rename(columns={"MOVIE_ID": "item_id"}, inplace=True)
    item_features_df["item_features"] = [
        item_features[i].tolist() for i in range(item_features.shape[0])
    ]

    return item_features_df


def prepareDouban(dataset_path_dict: dict, save_path: str) -> np.ndarray:
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
    rating_df = pd.read_csv(dataset_path_dict["rating"])
    # 读取用户信息
    user_df = pd.read_csv(dataset_path_dict["users"])
    # 读取项目信息
    item_df = pd.read_csv(dataset_path_dict["items"])

    # 去除无效评分
    movie_id = item_df["MOVIE_ID"].unique()
    rating_df = rating_df[rating_df["MOVIE_ID"].isin(movie_id)]

    # 划分测试集与训练集
    rating_df, train_data_df, val_data_df, test_data_df, uid_mapping, iid_mapping = handleRating(rating_df, save_path)
    # 获取用户特征
    user_features_df = doubanUserFeature(user_df)
    # 获取项目特征
    item_features_df = doubanItemFeature(item_df)

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
