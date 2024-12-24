#!/usr/bin/env python
# coding=utf-8
"""
@Description   LightGCN 训练
@Author        Alex_McAvoy
@Date          2024-10-14 15:15:05
"""

import pandas as pd
import matplotlib.pyplot as plt
import torch

from const import cfg
from utils.log import logger
from utils.wrapper import calTimes
from model.LightGCN.model import LightGCN
from model.LightGCN.loss import BPRLoss, sampleMiniBatch
from model.LightGCN.evaluation import calValLoss, getValRecommendations
from metrics.accurate import getAccurateMetrics
from metrics.diversity import getDiversityMetrics
from utils.graph import convertAdjMatrixToEdgeIndex
from utils.picture import plotMetric
from utils.trans import getInteractionMatrixByEdgeIndex, getItemDegreeByUserPosItemDict, getUserItemsDictByEdgeIndex


def getEmbeddingForBPR(model: LightGCN, user_num: int, item_num: int, 
                       train_edge_index: torch.Tensor, batch_size: int, device: torch.device) -> tuple:
    """
    @description 获取Embedding向量，以用于计算BPR损失
    @param {LightGCN} model LightGCN模型
    @param {int} user_num 用户数
    @param {int} item_num 项目数
    @param {torch} train_edge_index 训练邻接矩阵
    @param {int} batch_size mini_batch批次大小
    @param {torch} device 设备
    @return {tuple} 元组包含
            - {torch} users_emb_final 采样用户的最终Embedding向量
            - {torch} users_emb_0 采样用户的初始Embedding向量
            - {torch} pos_items_emb_final 采样正样本的最终Embedding向量
            - {torch} pos_items_emb_0 采样正样本的初始Embedding向量
            - {torch} neg_items_emb_final 采样负样本的最终Embedding向量
            - {torch} neg_items_emb_0 采样负样本的初始Embedding向量
    """
    # 前向传播
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(train_edge_index)
    
    # 将训练数据转回边索引
    edge_index_to_use = convertAdjMatrixToEdgeIndex(user_num, item_num, train_edge_index)
    
    # Mini-Batch采样，获取用户索引、正样本索引、负样本索引
    user_indices, pos_item_indices, neg_item_indices = sampleMiniBatch(batch_size, edge_index_to_use)
    user_indices, pos_item_indices, neg_item_indices = user_indices.to(device), pos_item_indices.to(device), neg_item_indices.to(device)
    
    # 获取采样用户、正样本、负样本对应的初始Embedding向量和最终Embedding向量
    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]
    
    return users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0

@calTimes(logger, "模型训练完成")
def trainLightGCN(user_num: int, item_num: int, edge_index: torch.Tensor, 
                  train_edge_index: torch.Tensor, val_edge_index: torch.Tensor) -> LightGCN:
    """
    @description 训练LightGCN
    @param {int} user_num 用户数
    @param {int} item_num 项目数
    @param {torch} edge_index 所有的边索引
    @param {torch} train_edge_index 训练邻接矩阵
    @param {torch} val_edge_index 测试邻接矩阵
    @return {LightGCN} 训练好的LightGCN模型
    """
    # 超参
    seed = cfg.MODEL["HyperParameter"]["seed"]                             # 种子值
    embedding_dim = cfg.MODEL["HyperParameter"]["embedding_dim"]           # Embedding维度
    layers = cfg.MODEL["HyperParameter"]["layers"]                         # 层数
    lr = cfg.MODEL["HyperParameter"]["lr"]                                 # 学习率
    gamma = cfg.MODEL["HyperParameter"]["gamma"]                           # 学习率衰减因子
    epochs = cfg.MODEL["HyperParameter"]["epochs"]                         # 训练轮次
    epoch_per_eval = cfg.MODEL["HyperParameter"]["epoch_per_eval"]         # 每多少epoch评估一次模型
    epoch_per_lr_decay = cfg.MODEL["HyperParameter"]["epoch_per_lr_decay"] # 每多少epoch衰减一次学习率
    batch_size = cfg.MODEL["HyperParameter"]["batch_size"]                 # Mini-batch批次大小
    epsilon_val = cfg.MODEL["HyperParameter"]["epsilon"]                   # 控制BPR损失L2正则化强度
    k = cfg.RECOMMEND["k"]                                                 # 推荐列表大小

    # 获取设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备：{device}")

    # 建立LightGCN模型
    torch.manual_seed(seed)
    model = LightGCN(user_num, item_num, embedding_dim, layers)

    # 把模型和数据移动到设备上
    model = model.to(device)
    edge_index = edge_index.to(device)
    train_edge_index = train_edge_index.to(device)
    val_edge_index = val_edge_index.to(device)

    # 设置模型为训练模式
    model.train()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    train_loss_ls = []
    val_loss_ls = []
    val_precision_ls = []
    val_recall_ls = []
    val_f1_ls = []
    val_ndcg_ls = []
    val_H_ls = []
    val_I_ls = []

    # 用户的正样本项目列表
    train_user_pos_items_dict = getUserItemsDictByEdgeIndex(convertAdjMatrixToEdgeIndex(user_num, item_num, train_edge_index))
    val_user_pos_items_dict = getUserItemsDictByEdgeIndex(convertAdjMatrixToEdgeIndex(user_num, item_num, val_edge_index))
    # 计算物品的度
    train_item_degree_dict = getItemDegreeByUserPosItemDict(train_user_pos_items_dict)
    # 计算交互矩阵
    train_interaction_mat = getInteractionMatrixByEdgeIndex(user_num, item_num, convertAdjMatrixToEdgeIndex(user_num, item_num, train_edge_index))

    # 训练
    for epoch in range(epochs):
        # 获取采样的用户、正样本、负样本的最终Embedding向量和初始Embedding向量
        (users_emb_final, users_emb_0, 
         pos_items_emb_final, pos_items_emb_0, 
         neg_items_emb_final, neg_items_emb_0) = getEmbeddingForBPR(model, user_num, item_num, 
                                                                    train_edge_index, batch_size, 
                                                                    device)

        # 计算损失
        train_loss = BPRLoss(users_emb_final, users_emb_0, 
                             pos_items_emb_final, pos_items_emb_0, 
                             neg_items_emb_final, neg_items_emb_0, 
                             epsilon_val)
        
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        train_loss.backward()
        # 更新模型参数
        optimizer.step()

        # 模型验证
        if epoch % epoch_per_eval == 0:
            # 设置模型为评估模式
            model.eval()

            with torch.no_grad():
                # 计算验证损失
                val_loss = calValLoss(model, user_num, item_num, val_edge_index, epsilon_val)

                # 计算验证阶段的用户推荐列表
                recommendations = getValRecommendations(model, user_num, item_num, train_edge_index, val_edge_index, k)
                # 评估准确性相关指标
                val_precision, val_recall, val_f1, val_ndcg = getAccurateMetrics(val_user_pos_items_dict, recommendations, k)
                # 评估多样性相关指标
                val_H, val_I = getDiversityMetrics(recommendations, train_item_degree_dict, train_interaction_mat, k)
                
                train_loss_ls.append(round(train_loss.item(), 5))
                val_loss_ls.append(val_loss)
                val_precision_ls.append(val_precision)
                val_recall_ls.append(val_recall)
                val_f1_ls.append(val_f1)
                val_ndcg_ls.append(val_ndcg)
                val_H_ls.append(val_H)
                val_I_ls.append(val_I)

                logger.info(f"[Iteration {epoch}/{epochs}]" + 
                            f"train_loss: {round(train_loss.item(), 5)}, val_loss: {val_loss}, val_precision@{k}: {val_precision}, " +
                            f"val_recall@{k}: {val_recall}, val_f1@{k}: {val_f1}, val_NDCG@{k}: {val_ndcg}, " + 
                            f"val_H@{k}: {val_H}, val_I@{k}: {val_I}")
            
            # 评估完毕后设置为训练模式
            model.train()

        # 学习率衰减
        if epoch % epoch_per_lr_decay == 0 and epoch != 0:
            scheduler.step()
    
    # 保存模型
    torch.save(model, cfg.MODEL["save_path"] + str(k) + "_LightGCN.pth")

    # 训练轮次列表
    iters = [epoch * epoch_per_eval for epoch in range(len(train_loss_ls))]
    
    # 构建DataFrame并保存
    save_path = cfg.PICTURES["save_path"] + "LightGCN_" + str(k) 
    val_metrics = pd.DataFrame({
        "iters": iters,
        "train_loss": train_loss_ls,
        "val_loss": val_loss_ls,
        "val_precision": val_precision_ls,
        "val_recall": val_recall_ls,
        "val_f1": val_f1_ls,
        "val_ndcg": val_ndcg_ls,
        "val_H": val_H_ls,
        "val_I": val_I_ls
    })
    val_metrics.to_csv(save_path + "_val_metrics.csv", index=False)

    # 绘制训练损失-验证损失图
    fig = plt.figure()
    plt.plot(iters, train_loss_ls, label="train")
    plt.plot(iters, val_loss_ls, label="validation")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("training and validation loss curves")
    plt.legend()
    plt.savefig(save_path + "_loss_curves.png")

    # 绘制模型训练过程中的验证集准确性指标图
    plotMetric(iters, val_precision_ls, "iteration", "precision", "precision curves", save_path + "_precision.png")
    plotMetric(iters, val_recall_ls, "iteration", "recall", "recall curves", save_path + "_recall.png")
    plotMetric(iters, val_f1_ls, "iteration", "F1-score", "F1-score curves", save_path + "_F1-score.png")
    plotMetric(iters, val_ndcg_ls, "iteration", "NDCG", "NDCG curves", save_path + "_NDCG.png")
    # 绘制模型训练过程中的验证集多样性指标图
    plotMetric(iters, val_H_ls, "iteration", "H", "H curves", save_path + "_H.png")
    plotMetric(iters, val_I_ls, "iteration", "I", "I curves", save_path + "_I.png")

    return model