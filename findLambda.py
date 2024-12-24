#!/usr/bin/env python
# coding=utf-8
"""
@Description   寻找最佳Lambda值
@Author        Alex_McAvoy
@Date          2024-10-17 17:59:54
"""

import numpy as np
import pandas as pd
import torch

from const import cfg
from utils.log import logger
from model.SpreadMethod.model import getSpreadingGeneralMat
from model.SpreadMethod.recommend import recommendSpreadMethod
from model.SpreadLightGCNOpti.model import getAllocateMat, getHybridSResourceMat
from model.SpreadLightGCNOpti.recommend import recommendForAllUser
from utils.trans import getInteractionMatrixByDataframe, recommendDictToTensor, getUserItemsDictByDataframe, getItemDegreeByUserPosItemDict
from metrics.accurate import getAccurateMetrics
from metrics.diversity import getDiversityMetrics
from utils.picture import plotMetric


def evaluation(test_user_pos_items_dict: dict, item_degree_dict: dict, 
               interaction_mat: np.ndarray, recommendations: torch.Tensor, 
               k: int):
    """
    @description 评估推荐结果
    @param {dict} test_user_pos_items_dict 测试集的用户正样本字典
    @param {dict} item_degree_dict 物品的度
    @param {np} interaction_mat 用户物品的交互矩阵
    @param {torch} recommendations 所有用户的TopK推荐
    @param {int} k 推荐列表大小
    @return {tuple} 元组包含
            - {float} precision 精确率
            - {float} recall 召回率
            - {float} f1 F1得分
            - {float} ndcg 归一化折损累积增益
            - {float} H 海明距离
            - {float} I 内部相似性
    """
    # 利用测试数据评估准确性相关指标
    precision, recall, f1, ndcg = getAccurateMetrics(test_user_pos_items_dict, recommendations, k)
    # 利用测试数据评估多样性相关指标
    H, I = getDiversityMetrics(recommendations, item_degree_dict, interaction_mat, k)
    return precision, recall, f1, ndcg, H, I

if __name__ == "__main__":
    # 过滤后的评分数据
    rating_df = pd.read_csv(cfg.PREPROCESSING["save_path"] + "filter_rating.csv")
    # 训练集
    train_data_df = pd.read_csv(cfg.PREPROCESSING["save_path"] + "train_data.csv")
    # 验证集
    val_data_df = pd.read_csv(cfg.PREPROCESSING["save_path"] + "val_data.csv")
    # 测试集
    test_data_df = pd.read_csv(cfg.PREPROCESSING["save_path"] + "test_data.csv")
    # 用户特征
    user_features_df = pd.read_csv(cfg.PREPROCESSING["save_path"] + "user_features.csv", sep="\t")
    # 项目特征
    item_features_df = pd.read_csv(cfg.PREPROCESSING["save_path"] + "item_features.csv", sep="\t")
    # 用户数
    user_num = len(rating_df['user_id'].unique())
    # 项目数
    item_num = len(rating_df['item_id'].unique())

    # 获取正样本字典
    train_user_pos_items_dict = getUserItemsDictByDataframe(train_data_df)
    val_user_pos_items_dict = getUserItemsDictByDataframe(val_data_df)
    test_user_pos_items_dict = getUserItemsDictByDataframe(test_data_df)
    # 计算物品的度
    item_degree_dict = getItemDegreeByUserPosItemDict(train_user_pos_items_dict, val_user_pos_items_dict)
    # 根据训练集和测试集，获取用户-项目交互矩阵
    A = getInteractionMatrixByDataframe(user_num, item_num, pd.concat([train_data_df, val_data_df]))

    # 推荐列表大小
    k = cfg.RECOMMEND["k"]
    # 根据LightGCNOpti模型，获取分配权重矩阵
    G = getAllocateMat(user_num, item_num, rating_df, train_data_df, val_data_df, test_data_df, user_features_df, item_features_df, k)
    # 计算通用扩散矩阵
    general_W = getSpreadingGeneralMat(A)

    lambda_val_ls = np.arange(0, 1 + 0.01, 0.01).tolist()
    # lambda_val_ls = np.arange(0, 1 + 0.1, 0.1).tolist()
    precision_ls = []
    recall_ls = []
    f1_ls = []
    ndcg_ls = []
    H_ls = []
    I_ls = []

    # 从0到1，步长为0.01设置lambda
    for lambda_val in lambda_val_ls:
        # 根据HybridS模型，获取资源扩散矩阵
        F = getHybridSResourceMat(A, general_W, lambda_val)
        # 计算分配权重矩阵与资源扩散矩阵的哈达玛积
        F_new = G * F
        # 获取推荐字典，SpreadLightGCN
        all_user_recommend_dict = recommendForAllUser(F_new, user_num, train_data_df, val_data_df, k)
        # 获取推荐字典，HybridS
        # all_user_recommend_dict = recommendSpreadMethod(user_num, item_num, train_data_df, val_data_df, "HybridS", lambda_val)

        # 将所有用户的推荐字典，转为tensor格式，以便于评估
        recommendations = recommendDictToTensor(all_user_recommend_dict)
        # 评估该轮推荐结果
        precision, recall, f1, ndcg, H, I = evaluation(test_user_pos_items_dict, item_degree_dict, A, recommendations, k)

        # 存储到相应结果列表中
        precision_ls.append(precision)
        recall_ls.append(recall)
        f1_ls.append(f1)
        ndcg_ls.append(ndcg)
        H_ls.append(H)
        I_ls.append(I)

        logger.info(f"Lambda: {lambda_val} 已评估完成")

    # 构建DataFrame并保存
    save_path = cfg.PICTURES["save_path"] + str(k) 
    val_metrics = pd.DataFrame({
        "lambda": lambda_val_ls,
        "precision": precision_ls,
        "recall": recall_ls,
        "f1": f1_ls,
        "ndcg": ndcg_ls,
        "H": H_ls,
        "I": I_ls
    })
    val_metrics.to_csv(cfg.EVALUATION["save_path"] + "lambda_evaluation_" + str(k) +".csv", index=False)

    # 绘制模型训练过程中的验证集准确性指标图
    plotMetric(lambda_val_ls, precision_ls, "lambda", "precision", "precision curves", cfg.EVALUATION["save_path"] + "precision_"+ str(k) + ".png")
    plotMetric(lambda_val_ls, recall_ls, "lambda", "recall", "recall curves", cfg.EVALUATION["save_path"] + "recall_" + str(k) + ".png")
    plotMetric(lambda_val_ls, f1_ls, "lambda", "F1-score", "F1-score curves", cfg.EVALUATION["save_path"] + "F1-score_" + str(k) + ".png")
    plotMetric(lambda_val_ls, ndcg_ls, "lambda", "NDCG", "NDCG curves", cfg.EVALUATION["save_path"] + "NDCG_" + str(k) + ".png")
    # 绘制模型训练过程中的验证集多样性指标图
    plotMetric(lambda_val_ls, H_ls, "lambda", "H", "H curves", cfg.EVALUATION["save_path"] + "H_" + str(k) + ".png")
    plotMetric(lambda_val_ls, I_ls, "lambda", "I", "I curves", cfg.EVALUATION["save_path"] + "I_" + str(k) + ".png")
