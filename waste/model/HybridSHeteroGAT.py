#!/usr/bin/env python
# coding=utf-8
"""
@Description   HyrbidS 与 异构GAT 混合推荐
@Author        Alex_McAvoy
@Date          2024-10-13 21:08:11
"""


import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv

from const import cfg
from utils.log import logger
from utils.wrapper import calTimes
from waste.processing.dataset import Dataset


class HeteroGAT(torch.nn.Module):
    def __init__(self, user_features_size: int, item_features_size: int, hidden_size: int) -> None:
        """
        @description HeteroGAT
        @param {*} self 类实例化对象
        @param {int} user_features_size 用户特征向量维度
        @param {int} item_features_size 项目特征向量维度
        @param {int} hidden_size 输出层维度
        """
        super(HeteroGAT, self).__init__()
        self.hetero_conv = HeteroConv(
            {
                ("users", "edge", "items"): GATConv(
                    user_features_size, hidden_size, add_self_loops=False
                ),
                ("items", "edge", "users"): GATConv(
                    item_features_size, hidden_size, add_self_loops=False
                ),
            }
        )

        self.user_linear = torch.nn.Linear(hidden_size, user_features_size)
        self.item_linear = torch.nn.Linear(hidden_size, item_features_size)

    def forward(self, data) -> tuple:
        """
        @description 前向传播
        @param {*} self 类实例化对象
        @param {HeteroData} data HeteroGAT异构图
        @return {tuple} 元组包含
                - {torch} user_out 输出用户特征向量
                - {torch} item_out 输出项目特征向量
        """
        out = self.hetero_conv(data.x_dict, data.edge_index_dict)
        user_out = self.item_linear(out["users"])
        item_out = self.user_linear(out["items"])

        return user_out, item_out


def buildGraph(dataset: Dataset) -> HeteroData:
    """
    @description 建立HeteroGAT图
    @param {Dataset} dataset 内部数据集
    @return {HeteroData} HeteroGAT图
    """
    # 特征向量
    user_features = torch.from_numpy(dataset.users_features_np).to(torch.float32)
    item_features = torch.from_numpy(dataset.items_features_np).to(torch.float32)
    # 特征向量维度
    user_features_size = user_features.shape[1]
    item_features_size = item_features.shape[1]


    # 定义元数据字典，描述异构图的结构
    num_users = dataset.data_df["user_id"].nunique()
    num_items = dataset.data_df["item_id"].nunique() 
    users_node = torch.tensor(dataset.data_df["user_id"].values, dtype=torch.long)
    items_node = torch.tensor(dataset.data_df["item_id"].values, dtype=torch.long)
    edge_index1 = torch.stack([users_node, items_node], dim=0)
    edge_index2 = torch.stack([items_node, users_node], dim=0)
    meta_dict = {
        'users': {'nodes_num': num_users, 'features_size': user_features_size},
        'items': {'nodes_num': num_items, 'features_size': item_features_size},
        ('users','edge','items'): {'edge_index': edge_index1},
        ('users','edge','items'): {'edge_index': edge_index2}
    }

    # 创建异构图数据对象
    graph = HeteroData(meta_dict)

    # 将节点特征和边索引添加到异构图对象中
    graph['users'].x = user_features
    graph['items'].x = item_features
    graph[('users','edge','items')].edge_index = edge_index1
    graph[('items','edge','users')].edge_index = edge_index2

    logger.info(f"HeteroData：{graph}")

    return graph

@calTimes(logger, "模型训练完成")
def trainHeteroGAT(graph: HeteroData) -> HeteroGAT:
    """
    @description HeteroGAT训练
    @param {Data} graph HeteroGAT图
    @return {HeteroGAT} HeteroGAT模型
    """

    # 超参
    user_features_size = graph.x_dict["users"].shape[1]  # 用户特征向量维度
    item_features_size = graph.x_dict["items"].shape[1]  # 项目特征向量维度
    hidden_dim = cfg.MODEL["hidden_dim"]                 # 隐藏层维度
    seed = cfg.MODEL["seed"]                             # 种子值
    lr = cfg.MODEL["lr"]                                 # 学习率
    epochs = cfg.MODEL["epochs"]                         # 训练轮次

    # 建立HeteroGAT模型
    torch.manual_seed(seed)
    model = HeteroGAT(user_features_size, item_features_size, hidden_dim)

    # 设置模型为训练模式
    model.train()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练
    for epoch in range(epochs):
        # 梯度清零
        optimizer.zero_grad()
        # 获取预测结果
        user_out, item_out = model(graph)
        # 计算预测结果与真实值的MSE
        loss = F.mse_loss(torch.cat([graph.x_dict["users"], graph.x_dict["items"]], dim=0), 
                          torch.cat([user_out, item_out], dim=0))
        # 反向传播
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 输出每轮损失
        logger.info(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    return model

@calTimes(logger, "资源分配权重矩阵计算完成")
def getHeteroGATMat(model: HeteroGAT, graph: HeteroData) -> torch.Tensor:
    """
    @description 评估HeteroGAT模型
    @param {HeteroGAT} model HeteroGAT模型
    @param {HeteroData} graph HeteroGAT图
    @param {int} user_num 用户数
    @param {int} k 推荐候选集大小
    @return {np} 资源分配权重矩阵
    """
    # 设置模型为评估模式
    model.eval()
    with torch.no_grad():
        # 获取模型预测的特征向量
        user_embedding, item_embedding = model(graph)
        # 资源分配权重矩阵
        G = torch.mm(user_embedding, item_embedding.T)

    return G.numpy()

@calTimes(logger, "通用扩散矩阵计算完成")
def getSpreadingGeneralMat(A: np.ndarray) -> np.ndarray:
    """
    @description 获取通用扩散矩阵
    @param {np} A 用户-物品交互矩阵
    @return {np} 通用扩散矩阵，用于后续物质扩散、热扩散、混合扩散
    """
    # 计算用户的度：每个用户交互过多少物品
    user_degrees = np.sum(A, axis=1)
    # 计算通用转移矩阵
    general_W = np.dot(A.T / user_degrees, A)

    return general_W
    
@calTimes(logger, "扩散资源矩阵矩阵完成")
def Hybrid(A: np.ndarray, general_W: np.ndarray, Lambda: float = 0.5) -> np.ndarray:
    """
    @description 概率扩散与热扩散混合推荐
    @param {np} A 用户-物品交互矩阵
    @param {np} general_W 通用转移矩阵
    @param {float} Lambda 混合算法比例常数，Lambda=0时，Hybrid退化为HeatS，Lambda=1时，Hybrid退化为ProbS
    @return {np} 转移矩阵，W[i][j]为物品i到物品j的转移值
    """
    # 计算物品的度：每个物品被多少用户交互
    item_degrees = np.sum(A, axis=0)

    # 计算物品度的混合幂次
    degree_alpha = np.power(item_degrees, 1 - Lambda)
    degree_beta = np.power(item_degrees, Lambda)
    
    # 计算转移矩阵
    W = general_W / (degree_alpha[:, np.newaxis] * degree_beta[np.newaxis, :])

    return W

def getResource(A: np.ndarray, W: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    @description 获取扩散后的目标用户对应的物品资源值
    @param {np} A 用户-物品交互矩阵
    @param {np} W 转移矩阵
    @param {np} G 权重矩阵
    @return {np} 扩散后的物品资源值，f_new[i]为用户i经过两轮扩散后的各物品的资源向量
    """
    # 初始资源向量：已交互过为1，未交互过为0
    f0 = A
    # 扩散
    f_new = np.dot(f0, W) * G
    return f_new

def recommendForAllUser(train_dataset: Dataset, A: np.ndarray, f_new: np.ndarray, k: int) -> dict:
    """
    @description 获取目标用户推荐列表
    @param {Dataset} train_dataset 内部数据集
    @param {np} A 用户-物品交互矩阵
    @param {np} f_new 扩散后的各用户物品资源向量
    @param {int} k 推荐列表大小
    @return {dict} 所有用户的推荐列表
    """
    # 枚举所有用户内部id
    result = dict()
    for uIdxin in train_dataset.uIdx_np:
        # 获取当前用户的未交互物品集
        non_interacted_items = np.where(A[uIdxin] == 0)[0]

        # 获取当前用户未交互过的物品资源值列表
        non_interacted_items_values = f_new[uIdxin][non_interacted_items]

        # 打包成字典，键：物品id，值：物品资源值
        zipped_dict = dict(zip(non_interacted_items,non_interacted_items_values))

        # 按照资源值降序排序
        sorted_zipped_dict = dict(sorted(zipped_dict.items(), key=lambda item: item[1], reverse=True))

        # 取前 k 个作为推荐列表
        candidate_list = list(dict(list(sorted_zipped_dict.items())[:k]).keys())

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
def recommendHybridSHeteroGAT(train_dataset: Dataset, A: np.ndarray, k: int = cfg.RECOMMEND["k"]) -> dict:
    """
    @description HeteroGAT推荐
    @param {Dataset} train_dataset 内部数据集
    @param {np} A 用户-物品交互矩阵
    @param {int} k 目标用户推荐候选集大小
    @return {dict} 所有用户推荐候选集
    """

    # 构建HeteroGAT图
    logger.info("正在建立HeteroGAT图")
    graph = buildGraph(train_dataset)
    logger.info("-------------")

    # 训练
    logger.info("正在训练GAT模型")
    model = trainHeteroGAT(graph)
    logger.info("-------------")

    # 评估模型，获取资源分配权重矩阵
    logger.info("正在计算相关矩阵")
    G = getHeteroGATMat(model, graph)
    # 计算通用扩散矩阵
    general_W = getSpreadingGeneralMat(A)
    # 获取转移矩阵
    W = Hybrid(A, general_W, cfg.MODEL["Lambda"])
    # 获取扩散后的资源值
    f_new = getResource(A, W, G)
    logger.info("-------------")

   

    # 推荐
    logger.info("正在进行推荐")
    return recommendForAllUser(train_dataset, A, f_new, k)
    
