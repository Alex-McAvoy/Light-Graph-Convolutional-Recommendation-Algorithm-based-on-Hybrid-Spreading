#!/usr/bin/env python
# coding=utf-8
"""
@Description   推荐主函数
@Author        Alex_McAvoy
@Date          2024-09-18 20:26:41
"""
import numpy as np
import pandas as pd

from const import cfg
from utils.log import logger
from processing.handleMovielens import prepareMovieLens
from processing.handleDouban import prepareDouban

from model.SpreadMethod.recommend import recommendSpreadMethod
from model.LightGCN.recommend import recommendLightGCN
from model.LightGCNOpti.recommend import recommendLightGCNOpti
from model.SpreadLightGCN.recommend import recommendSpreadLightGCN
from model.SpreadLightGCNOpti.recommend import recommendSpreadLightGCNOpti
from utils.trans import getInteractionMatrixByDataframe, getUserItemsDictByDataframe, recommendDictToTensor, getItemDegreeByUserPosItemDict
from metrics.accurate import getAccurateMetrics
from metrics.diversity import getDiversityMetrics

if __name__ == "__main__":

    logger.info("Step1：正在加载预处理数据")
    try:
        # 过滤后的评分数据
        rating_df = pd.read_csv(cfg.PREPROCESSING["save_path"] + "filter_rating.csv")
        # 训练集
        train_data_df = pd.read_csv(cfg.PREPROCESSING["save_path"] + "train_data.csv")
        # 验证集
        val_data_df = pd.read_csv(cfg.PREPROCESSING["save_path"] + "val_data.csv")
        # 测试集
        test_data_df = pd.read_csv(cfg.PREPROCESSING["save_path"] + "test_data.csv")
        # 用户特征向量
        user_features_df = pd.read_csv(cfg.PREPROCESSING["save_path"] + "user_features.csv", sep="\t")
        # 项目特征向量
        item_features_df = pd.read_csv(cfg.PREPROCESSING["save_path"] + "item_features.csv", sep="\t")
    except:
        logger.info("预处理数据读取失败，正在重新计算")
        if cfg.DATA_SET == "movielens":
            rating_df, train_data_df, val_data_df, test_data_df, user_features_df, item_features_df = prepareMovieLens(cfg.PREPROCESSING["dataset_path_dict"], cfg.PREPROCESSING["save_path"])
        elif cfg.DATA_SET == "douban":
            rating_df, train_data_df, val_data_df, test_data_df, user_features_df, item_features_df = prepareDouban(cfg.PREPROCESSING["dataset_path_dict"], cfg.PREPROCESSING["save_path"] )
    # 用户数
    user_num = len(rating_df['user_id'].unique())
    # 项目数
    item_num = len(rating_df['item_id'].unique())
    logger.info(f"总用户数：{user_num}，总项目数：{item_num}")
    logger.info(f"训练集 ：{train_data_df.shape}")
    logger.info(f"验证集 ：{val_data_df.shape}")
    logger.info(f"测试集 ：{test_data_df.shape}")
    logger.info(f"用户特征向量 shape: ({user_features_df.shape[0]}, {len(user_features_df['user_features'].iloc[0])})")
    logger.info(f"物品特征向量 shape: ({item_features_df.shape[0]}, {len(item_features_df['item_features'].iloc[0])})")
    logger.info("预处理数据加载完毕")
    logger.info("-------------------------------------------------------")

    logger.info(f"Step2：正在读取推荐结果")
    try:
        all_user_recommend_dict = np.load(cfg.RECOMMEND["save_path"] + "all_user_recommend_dict_"  + cfg.MODEL["name"] + str(cfg.RECOMMEND["k"]) + ".npy", allow_pickle=True).item()
        logger.info("推荐结果读取完毕")
    except:
        logger.info(f"推荐结果读取失败，正在重新进行推荐，选用模型：{cfg.MODEL['name']}")
        # 根据选择的模型进行推荐
        if cfg.MODEL["name"] == "ProbS":
            all_user_recommend_dict = recommendSpreadMethod(user_num, item_num, train_data_df, val_data_df, "ProbS")
        elif cfg.MODEL["name"] == "HeatS":
            all_user_recommend_dict = recommendSpreadMethod(user_num, item_num, train_data_df, val_data_df, "HeatS")
        elif cfg.MODEL["name"] == "HybridS":
            all_user_recommend_dict = recommendSpreadMethod(user_num, item_num, train_data_df, val_data_df, "HybridS")
        elif cfg.MODEL["name"] == "LightGCN":
            all_user_recommend_dict = recommendLightGCN(user_num, item_num, rating_df, train_data_df, val_data_df, test_data_df)
        elif cfg.MODEL["name"] == "LightGCNOpti":
            all_user_recommend_dict = recommendLightGCNOpti(user_num, item_num, rating_df, train_data_df, val_data_df, test_data_df, user_features_df, item_features_df)
        elif cfg.MODEL["name"] == "SpreadLightGCN":
            all_user_recommend_dict = recommendSpreadLightGCN(user_num, item_num, rating_df, train_data_df, val_data_df, test_data_df)
        elif cfg.MODEL["name"] == "SpreadLightGCNOpti":
            all_user_recommend_dict = recommendSpreadLightGCNOpti(user_num, item_num, rating_df, train_data_df, val_data_df, test_data_df, user_features_df, item_features_df)
    logger.info("-------------------------------------------------------")

    logger.info(f"Step3：正在评估推荐结果")

    # 将所有用户的推荐字典，转为tensor格式，以便于评估
    recommendations = recommendDictToTensor(all_user_recommend_dict)
    # 获取正样本字典
    train_user_pos_items_dict = getUserItemsDictByDataframe(train_data_df)
    val_user_pos_items_dict = getUserItemsDictByDataframe(val_data_df)
    test_user_pos_items_dict = getUserItemsDictByDataframe(test_data_df)
    # 计算物品的度
    item_degree_dict = getItemDegreeByUserPosItemDict(train_user_pos_items_dict, val_user_pos_items_dict)
    # 计算交互矩阵
    interaction_mat = getInteractionMatrixByDataframe(user_num, item_num, pd.concat([train_data_df, val_data_df]))
    
    # 推荐列表大小
    k = cfg.RECOMMEND["k"]
    # 利用测试数据评估准确性相关指标
    test_precision, test_recall, test_f1, test_ndcg = getAccurateMetrics(test_user_pos_items_dict, recommendations, k)
    # 利用测试数据评估多样性相关指标
    test_H, test_I = getDiversityMetrics(recommendations, item_degree_dict, interaction_mat, k)

    logger.info(f"[{cfg.MODEL['name']} Test Accurate] " +
                f"precision@{k}: {test_precision}, recall@{k}: {test_recall}, f1@{k}: {test_f1}, " + 
                f"NDCG@{k}: {test_ndcg}")
    logger.info(f"[{cfg.MODEL['name']} Test Diversity] H@{k}: {test_H}, I@{k}: {test_I}")

    
    