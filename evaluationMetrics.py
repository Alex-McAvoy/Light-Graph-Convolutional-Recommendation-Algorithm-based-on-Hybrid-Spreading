#!/usr/bin/env python
# coding=utf-8
"""
@Description   推荐结果评估
@Author        Alex_McAvoy
@Date          2024-10-11 20:58:22
"""

import numpy as np
import pandas as pd

from utils.trans import getInteractionMatrixByDataframe, getUserItemsDictByDataframe, recommendDictToTensor, getItemDegreeByUserPosItemDict
from metrics.accurate import getAccurateMetrics
from metrics.diversity import getDiversityMetrics

from const import cfg
from utils.log import logger

if __name__ == "__main__":

    logger.info("Step1：正在加载预处理数据")
    # 训练集
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
    # 用户数
    user_num = len(rating_df['user_id'].unique())
    # 项目数
    item_num = len(rating_df['item_id'].unique())
    logger.info("预处理数据加载完毕")
    logger.info("-------------------------------------------------------")

    logger.info(f"Step2：正在评估 {cfg.DATA_SET} 数据集推荐结果")
    # 模型
    models = ["ProbS", "HeatS", "HybridS", "LightGCN", "SpreadLightGCN", "SpreadLightGCNOpti"]
    # 推荐列表长度
    recommend_len = [30, 50, 100]
    
    # 存储所有推荐列表长度k对应的Dataframe
    result_sheets = {}

    # 遍历所有推荐列表长度
    for k in recommend_len:
        # 存储k值下每个模型的指标
        results_k = []

        logger.info("---------------------------")
        # 遍历所有模型
        for model in models:
            # 模型model，推荐列表长度k下的，所有用户的推荐结果
            all_user_recommend_dict = np.load(cfg.RECOMMEND["save_path"] + "all_user_recommend_dict_"  + model + "_" + str(k) + ".npy", allow_pickle=True).item()
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

            # 利用测试数据评估准确性相关指标
            test_precision, test_recall, test_f1, test_ndcg = getAccurateMetrics(test_user_pos_items_dict, recommendations, k)
            # 利用测试数据评估多样性相关指标
            test_H, test_I = getDiversityMetrics(recommendations, item_degree_dict, interaction_mat, k)

            # 存储指标结果
            results_k.append({
                "Model": model, 
                "P": test_precision, 
                "R": test_recall, 
                "F1": test_f1, 
                "NDCG": test_ndcg, 
                "H": test_H,
                "I": test_I
            })
            logger.info("---------------")
            logger.info(f"推荐列表长度为：{k}，模型为：{model} ")
            logger.info(f"准确性指标：P={test_precision}, R={test_recall}, f1={test_f1}, NDCG={test_ndcg}")
            logger.info(f"多样性指标：H={test_H}, I={test_I}")
            logger.info("---------------")
        # 存储当前k值的评估结果
        result_sheets[k] = pd.DataFrame(results_k)
    # 存储结果
    with pd.ExcelWriter(cfg.EVALUATION["save_path"] + "model_evaluation_results.xlsx") as writer:
        for k, df in result_sheets.items():
            df.to_excel(writer, sheet_name=f"{k}", index=False)
    logger.info("---------------------------")
    logger.info("评估完成")
