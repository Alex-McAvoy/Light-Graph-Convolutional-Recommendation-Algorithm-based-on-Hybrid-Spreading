#!/usr/bin/env python
# coding=utf-8
"""
@Description   GNN推荐
@Author        Alex_McAvoy
@Date          2024-09-29 21:18:52
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from const import cfg
from utils.log import logger
from utils.wrapper import calTimes
from waste.processing.dataset import Dataset


class GNN(torch.nn.Module):
    def __init__(self, input_dim:int, hidden_dim: int, output_dim: int) -> None:
        """
        @description GNN
        @param {*} self 类实例化对象
        @param {*} input_dim 输入维度
        @param {*} hidden_dim 隐藏层维度
        @param {*} output_dim 输出维度
        """
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """
        @description 前向传播
        @param {*} self 类实例化对象
        @param {Data} data GNN图
        @return {torch} x 输出特征向量
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def buildGraph(dataset: Dataset) -> Data:
    """
    @description 建立GNN图
    @param {Dataset} dataset 内部数据集
    @return {Data} GNN图
    """

    # 边集，每条交互数据建立一条边
    users_node = torch.tensor(dataset.data_df["user_id"].values, dtype=torch.long)
    items_node = torch.tensor(dataset.data_df["item_id"].values, dtype=torch.long)
    edge_index = torch.stack([users_node, items_node + dataset.data_df["user_id"].nunique()], dim=0)

    # 合并特征
    features = np.vstack((dataset.users_features_np, dataset.items_features_np))
    features = torch.from_numpy(features)
    features = features.to(torch.float32)

    # 建图
    graph = Data(edge_index=edge_index)
    graph.x = features

    logger.info(f"边数：{edge_index.shape[1]}")
    logger.info(f"结点数：{graph.x.shape[0]}")
    logger.info(f"向量特征数：{graph.x.shape[1]}")

    return graph

@calTimes(logger, "模型训练完成")
def trainGNN(graph: Data) -> GNN:
    """
    @description GNN训练
    @param {Data} graph GNN图
    @return {GNN} GNN模型
    """
    
    # 超参
    node_num, input_dim = graph.x.shape    # 结点数、输入层维度 
    hidden_dim = cfg.MODEL["hidden_dim"]   # 隐藏层维度
    output_dim = input_dim               # 输出层维度
    seed = cfg.MODEL["seed"]             # 种子值
    lr = cfg.MODEL["lr"]                 # 学习率
    epochs = cfg.MODEL["epochs"]         # 训练轮次

    # 建立GNN模型
    torch.manual_seed(seed)
    model = GNN(input_dim, hidden_dim, output_dim)

    # 设置模型为训练模式
    model.train()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练
    for epoch in range(epochs):
        # 梯度清零
        optimizer.zero_grad()
        # 获取预测结果
        out = model(graph)
        # 计算预测结果与真实值的MSE
        loss = F.mse_loss(out, graph.x)
        # 反向传播
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 输出每轮损失
        logger.info(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    return model

def evalGNN(model: GNN, graph: Data, user_num: int, k: int) -> torch.Tensor:
    """
    @description 评估GNN模型
    @param {GNN} model GNN模型
    @param {Data} graph GNN图
    @param {int} user_num 用户数
    @param {int} k 推荐候选集大小
    @return {np} 所有用户的推荐候选集
    """
    # 设置模型为评估模式
    model.eval()
    with torch.no_grad():
        # 获取模型预测的特征向量
        embedding = model(graph)
        # 用户特征向量
        user_embedding = embedding[:user_num]
        # 物品特征向量
        item_embedding = embedding[user_num:]
        # 向量相乘计算得分
        scores = torch.mm(user_embedding, item_embedding.t())
        # 预测，考虑后续要筛出交互过的用户，k值扩大两倍
        _, recommendations = scores.topk(scores.shape[1], dim=1)
    
    return recommendations.numpy()

def recommendForAllUser(train_dataset: Dataset, model: GNN, graph: Data, k: int) -> np.ndarray:
    """
    @description 获取所有用户的推荐列表
    @param {Dataset} train_dataset 内部数据集
    @param {GNN} model GNN模型
    @param {Data} graph GNN图
    @param {int} k 推荐列表大小
    @return {np} 所有用户的推荐列表
    """

    # 用户数
    user_num = train_dataset.uIdx_np.shape[0]
    # 所有用户的推荐列表
    recommendation_np = evalGNN(model, graph, user_num, k)

    result = dict()
    # 枚举所有用户内部id
    for uIdxin in train_dataset.uIdx_np:
        # 当前用户交互列表
        interaction_list = train_dataset.data_df[train_dataset.data_df["user_id"] == uIdxin]["item_id"].to_numpy()
        # 当前用户推荐列表
        recommend_list = recommendation_np[uIdxin]
        # 候选集为二者的差集再取前k个
        candidate_index = np.in1d(recommend_list, interaction_list)
        candidate_list = recommend_list[~candidate_index][:k].tolist()

        # 用户内部id转为外部id
        uid = train_dataset.idx2uid_map[uIdxin]
        # 物品内部id转为外部id
        iids = list(map(train_dataset.idx2iid_map.get, candidate_list))
        # 存入字典
        result[uid] = iids

    # 存储推荐结果
    np.save(cfg.RECOMMEND["save_path"] + "all_user_recommend_dict_" + cfg.MODEL["name"] + "_" + str(cfg.RECOMMEND["k"]) + ".npy", result)

    return result

@calTimes(logger, "推荐完成")
def recommendGNN(train_dataset: Dataset, k:int = cfg.RECOMMEND["k"]) -> dict:
    """
    @description GNN推荐
    @param {Dataset} train_dataset 内部数据集
    @param {int} k 目标用户推荐候选集大小
    @return {dict} 所有用户推荐候选集
    """
    # 构建GNN图
    logger.info("正在建立GNN图")
    graph = buildGraph(train_dataset)
    logger.info("-------------")

    # 训练
    logger.info("正在训练模型")
    model = trainGNN(graph)
    logger.info("-------------")

    # 推荐
    logger.info("正在进行推荐")
    return recommendForAllUser(train_dataset, model, graph, k)