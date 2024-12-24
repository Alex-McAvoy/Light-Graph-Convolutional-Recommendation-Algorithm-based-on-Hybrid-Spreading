#!/usr/bin/env python
# coding=utf-8
"""
@Description   数据预处理通用函数
@Author        Alex_McAvoy
@Date          2024-09-26 22:01:45
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from const import cfg
from utils.log import logger
from utils.wrapper import calTimes

@calTimes(logger, "用户-项目评分数据处理完成")
def handleRating(rating_df: pd.DataFrame, save_path: str) -> tuple:
    """
    @description 预处理评分数据
    @param {pd} rating_df 评分数据
    @param {str} save_path 存储路径
    @param {float} train_percentage 训练集所占比例
    @param {int} random_state 随机种子
    @return {tuple} 元组包含
        - {pd} filtered_rating_df 清洗后的用户-评分数据
        - {pd} train_data_df 训练集数据
        - {pd} val_data_df 验证集数据
        - {pd} test_data_df 测试集数据
        - {dict} uid_mapping 用户id映射字典
        - {dict} iid_mapping 项目id映射字典
    """

    # 列名映射
    columns_map = cfg.PREPROCESSING["columns_map"]
    # 分位点
    quantile_dict = cfg.PREPROCESSING["quantile"]

    # 根据用户id计算每个用户的评分数量
    user_rating_counts = rating_df[columns_map["user_id"]].value_counts().reset_index()
    user_rating_counts.columns = ["user_id", "rating_num"]

    # 根据分位数设定阈值
    threshold_start = user_rating_counts["rating_num"].quantile(quantile_dict["start"])
    threshold_end = user_rating_counts["rating_num"].quantile(quantile_dict["end"])
    logger.info(f"上分位点 {quantile_dict['start']} 阈值：{threshold_start}")
    logger.info(f"下分位点 {quantile_dict['end']} 阈值：{threshold_end}")

    # 根据分位数设定的阈值，过滤用户
    filtered_users = user_rating_counts[
        (user_rating_counts["rating_num"] >= threshold_end)
        & (user_rating_counts["rating_num"] <= threshold_start)
    ]

    # 筛选评论次数超过20次的数据
    filtered_rating_df = rating_df[
        rating_df[columns_map["user_id"]].isin(filtered_users["user_id"])
    ]
    filtered_rating_df = pd.DataFrame(
        filtered_rating_df,
        columns=[
            columns_map["user_id"],
            columns_map["item_id"],
            columns_map["rating"],
            columns_map["rating_time"],
        ],
    )
    filtered_rating_df.columns = ["user_id", "item_id", "rating", "rating_time"]

    # 转为内部ID
    lbl_user = LabelEncoder()
    lbl_item = LabelEncoder()
    filtered_rating_df.user_id = lbl_user.fit_transform(filtered_rating_df.user_id.values)
    filtered_rating_df.item_id = lbl_item.fit_transform(filtered_rating_df.item_id.values)

    # 获取编码映射字典
    uid_mapping = dict(zip(lbl_user.classes_, lbl_user.transform(lbl_user.classes_)))
    iid_mapping = dict(zip(lbl_item.classes_, lbl_item.transform(lbl_item.classes_)))

    logger.info("评分数据预处理，正在划分数据集")

    filtered_rating_df.to_csv(save_path + "/filter_rating.csv", sep=",", index=False, header=True)
    filtered_rating_df.reset_index(drop=True, inplace=True)

    # 交互数
    interactions_num = filtered_rating_df.shape[0]
    
    # 按照索引进行划分
    all_indices = [i for i in range(interactions_num)]
    train_indices, test_indices = train_test_split(all_indices, 
                                                test_size=cfg.PREPROCESSING["split_percentage"][0], 
                                                random_state=cfg.PREPROCESSING["seed"])
    val_indices, test_indices = train_test_split(test_indices, 
                                                test_size=cfg.PREPROCESSING["split_percentage"][1], 
                                                random_state=cfg.PREPROCESSING["seed"])
    
    # 根据索引获取训练集、验证集、测试集
    train_data_df = filtered_rating_df.loc[train_indices]
    val_data_df = filtered_rating_df.loc[val_indices]
    test_data_df = filtered_rating_df.loc[test_indices]

    # 保存训练集
    logger.info("-----------------------------------------")
    train_data_df.to_csv(save_path + "/train_data.csv", sep=",", index=False, header=True)
    logger.info(f"训练集评分数：{train_data_df.shape[0]}")
    logger.info(f"训练集用户数：{train_data_df['user_id'].unique().shape[0]}")
    logger.info(f"训练集项目数：{train_data_df['item_id'].unique().shape[0]}")
    logger.info("-----------------------------------------")

    # 保存验证集
    val_data_df.to_csv(save_path + "/val_data.csv", sep=",", index=False, header=True)
    logger.info(f"验证集评分数：{val_data_df.shape[0]}")
    logger.info(f"验证集用户数：{val_data_df['user_id'].unique().shape[0]}")
    logger.info(f"验证集电影数：{val_data_df['item_id'].unique().shape[0]}")
    logger.info("-----------------------------------------")

    # 保存测试集
    test_data_df.to_csv(save_path + "/test_data.csv", sep=",", index=False, header=True)
    logger.info(f"测试集评分数：{test_data_df.shape[0]}")
    logger.info(f"测试集用户数：{test_data_df['user_id'].unique().shape[0]}")
    logger.info(f"测试集电影数：{test_data_df['item_id'].unique().shape[0]}")
    logger.info("-----------------------------------------")
    
    return filtered_rating_df, train_data_df, val_data_df, test_data_df, uid_mapping, iid_mapping