#!/usr/bin/env python
# coding=utf-8
"""
@Description   图处理工具
@Author        Alex_McAvoy
@Date          2024-10-14 15:39:13
"""
import torch
from torch_sparse import SparseTensor

# 转为邻接矩阵
def convertEdgeIndexToAdjMatrix(user_num: int, item_num: int, edge_index: torch.Tensor) -> torch.Tensor:
    """
    @description 将边索引转为稀疏邻接矩阵
    @param {int} user_num 用户数
    @param {int} item_num 物品数
    @param {torch} edge_index 边索引
            e.g. tensor([[412, 312, 591,  ...,  40, 377, 863],
                         [299, 518, 432,  ...,  97, 175,  49]])
    @return {torch} 邻接矩阵
    """
    R = torch.zeros((user_num, item_num))
    for i in range(len(edge_index[0])):
        row_idx = edge_index[0][i]
        col_idx = edge_index[1][i]
        R[row_idx][col_idx] = 1

    R_transpose = torch.transpose(R, 0, 1)
    adj_mat = torch.zeros((user_num + item_num , user_num + item_num))
    adj_mat[: user_num, user_num :] = R.clone()
    adj_mat[user_num :, : user_num] = R_transpose.clone()
    adj_mat_coo = adj_mat.to_sparse_coo()
    adj_mat_coo = adj_mat_coo.indices()

    return adj_mat_coo


def convertAdjMatrixToEdgeIndex(user_num: int, item_num: int, edge_index: torch.Tensor) -> torch.Tensor:
    """
    @description 将稀疏邻接矩阵转为边索引
    @param {int} user_num 用户数
    @param {int} item_num 项目数
    @param {torch} edge_index 稀疏邻接矩阵
    @return {torch} 边索引
    """
    sparse_input_edge_index = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=((user_num + item_num), user_num + item_num))
    adj_mat = sparse_input_edge_index.to_dense()
    interact_mat = adj_mat[: user_num, user_num :]
    r_mat_edge_index = interact_mat.to_sparse_coo().indices()
    return r_mat_edge_index