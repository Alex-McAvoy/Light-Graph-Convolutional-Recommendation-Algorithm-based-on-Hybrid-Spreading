#!/usr/bin/env python
# coding=utf-8
"""
@Description   配置文件
@Author        Alex_McAvoy
@Date          2024-09-18 20:28:20
"""
import os


class Config(object):
    def __init__(self, env: str, dataset: str, model: str) -> None:
        """
        @description 构造函数
        @param {*} self 类实例对象
        @param {str} env 环境
        @param {str} dataset 采用数据集
        @param {str} model 采用模型
        """
        # 开发环境
        if env == "dev":
            self._config = DevConfig(dataset,model)
        # 超算环境
        elif env == "prod":
            self._config = ProdConfig(dataset,model)

        self.mkdir()

    @property
    def config(self):
        return self._config

    def mkdir(self):
        if not os.path.exists(self._config.LOG["file_path"]):
            os.makedirs(self._config.LOG["file_path"])

        if not os.path.exists(self._config.PREPROCESSING["save_path"]):
            os.makedirs(self._config.PREPROCESSING["save_path"])

        if not os.path.exists(self._config.RECOMMEND["save_path"]):
            os.makedirs(self._config.RECOMMEND["save_path"])

        if not os.path.exists(self._config.MODEL["save_path"]):
            os.makedirs(self._config.MODEL["save_path"])

        if not os.path.exists(self._config.EVALUATION["save_path"]):
            os.makedirs(self._config.EVALUATION["save_path"])

        if not os.path.exists(self._config.PICTURES["save_path"]):
            os.makedirs(self._config.PICTURES["save_path"])
        
class DevConfig(object):
    """
    @description 开发环境配置
    """

    def __init__(self, dataset: str, model: str) -> None:
        """
        @description 构造函数
        @param {*} self 类实例化对象
        @param {str} dataset 采用数据集
        @param {str} model 采用模型
        """
        tmp_path = "./RS/algorithm/temp"

        # 采用数据集
        if dataset == "movielens":
            tmp_path += "/movielens"
            self.DATA_SET = "movielens"
        elif dataset == "douban":
            tmp_path += "/douban"
            self.DATA_SET = "douban"
        elif dataset == "netfilx":
            tmp_path += "/netfilx"
            self.DATA_SET = "netfilx"
        
        # 数据预处理配置
        self.PREPROCESSING = {
            "seed": 42,
            # 数据集存储路径
            "dataset_path_dict": {},
            # 预处理数据存储路径
            "save_path": tmp_path + "/preprocess/",
            # Word2Vec向量大小
            "vector_size": {},
            # 列名映射
            "columns_map": {},
            # 分位点
            "quantile": {
                "start": 1,
                "end": 0
            },
            # 训练集、验证集、测试集划分比例 8:1:1（0.2代表20%的验证集、测试集，0.5代表在20%中的验证集训练集中按照50%比例再进行划分）
            "split_percentage": [0.2, 0.5],
        }

        # 日志配置
        self.LOG = {"file_path": tmp_path + "/log/"}

        # 模型配置
        self.MODEL = {
            # 模型名
            "name": "",
            # 超参
            "HyperParameter": {},
            # 存储路径
            "save_path": tmp_path + "/model/"
        }

        # 模型配置
        if model == "ProbS":
            self.MODEL["name"] = "ProbS"
            self.MODEL["HyperParameter"] = {
                # 混合比例常数
                "lambda": 1
            }
        elif model == "HeatS":
            self.MODEL["name"] = "HeatS"
            self.MODEL["HyperParameter"] = {
                # 混合比例常数
                "lambda": 0
            }
        elif model == "HybridS":
            self.MODEL["name"] = "HybridS"
            self.MODEL["HyperParameter"] = {
                # 混合比例常数
                "lambda": 0.3
            }
        elif model == "LightGCN":
            self.MODEL["name"] = "LightGCN"
            self.MODEL["HyperParameter"] = {
                # 种子值
                "seed": 42,
                # Embedding维度
                "embedding_dim": 64,
                # GCN层数
                "layers": 3,
                # 学习率
                "lr": 1e-3,
                # 学习率衰减因子
                "gamma": 0.95,
                # 训练轮次
                "epochs": 10,
                # 每多少epoch评估一次模型
                "epoch_per_eval": 200,
                # 每多少epoch衰减一次学习率
                "epoch_per_lr_decay": 200,
                # Mini-batch批次大小
                "batch_size": 1024,
                # 控制BPR损失L2正则化强度
                "epsilon": 1e-6
            }
        elif model == "SpreadLightGCN":
            self.MODEL["name"] = "SpreadLightGCN"
            self.MODEL["HyperParameter"] = {
                # 种子值
                "seed": 42,
                # Embedding维度
                "embedding_dim": 64,
                # GCN层数
                "layers": 3,
                # 学习率
                "lr": 1e-3,
                # 学习率衰减因子
                "gamma": 0.95,
                # 训练轮次
                "epochs": 10,
                # 每多少epoch评估一次模型
                "epoch_per_eval": 200,
                # 每多少epoch衰减一次学习率
                "epoch_per_lr_decay": 200,
                # Mini-batch批次大小
                "batch_size": 1024,
                # 控制BPR损失L2正则化强度
                "epsilon": 1e-6,
                # 混合比例常数
                "lambda": 0.5
            }

        # 评估配置
        self.EVALUATION = {
            # 存储路径
            "save_path": tmp_path + "/evaluation/"
        }

        # 推荐配置
        self.RECOMMEND = {
            # 推荐候选集大小
            "k": 10,
            # 存储路径
            "save_path": tmp_path + "/recommend/"
        }

        # 图片配置
        self.PICTURES = {
            # 存储路径
            "save_path": tmp_path + "/pictures/"
        }

        # MovieLens数据集预处理参数设置
        if dataset == "movielens":
            self.PREPROCESSING["dataset_path_dict"] = {
                "users": "I:/Workspace/data/ml-100k/u.user",
                "items": "I:/Workspace/data/ml-100k/u.item",
                "rating": "I:/Workspace/data/ml-100k/u.data",
                "occupation": "I:/Workspace/data/ml-100k/u.occupation"
            }
            self.PREPROCESSING["columns_map"] = {
                "user_id": "user",
                "item_id": "item",
                "rating": "rating",
                "rating_time": "timestamp"
            }
            self.PREPROCESSING["quantile"] = {
                "start": 1,
                "end": 0
            }
            self.PREPROCESSING["vector_size"] = {
                "title": 5,
                "content": 20
            }
        # douban数据集预处理参数设置
        elif dataset == "douban":
            self.PREPROCESSING["dataset_path_dict"] = {
                "users":"I:/Workspace/data/douban/users.csv",
                "items":"I:/Workspace/data/douban/movies.csv",
                "rating":"I:/Workspace/data/douban/ratings.csv"
            }
            self.PREPROCESSING["columns_map"] = {
                "user_id": "USER_MD5",
                "item_id": "MOVIE_ID",
                "rating": "RATING",
                "rating_time": "RATING_TIME"
            }
            self.PREPROCESSING["quantile"] = {
                "start": 0.991,
                "end": 0.99
            }
            self.PREPROCESSING["vector_size"] = {
                "title": 3,
                "content": 20
            }

            self.RECOMMEND["target_user"] = "1a76e2591cd2f3740ccb7f198dace22a"
        
class ProdConfig(object):
    """
    @description 开发环境配置
    """

    def __init__(self, dataset: str, model: str) -> None:
        """
        @description 构造函数
        @param {*} self 类实例化对象
        @param {str} dataset 采用数据集
        @param {str} model 采用模型
        """
        tmp_path = "/data/alex/algorithm"

        # 采用数据集
        if dataset == "movielens":
            tmp_path += "/movielens"
            self.DATA_SET = "movielens"
        elif dataset == "douban":
            tmp_path += "/douban"
            self.DATA_SET = "douban"
        elif dataset == "netfilx":
            tmp_path += "/netfilx"
            self.DATA_SET = "netfilx"
        
        # 数据预处理配置
        self.PREPROCESSING = {
            "seed": 42,
            # 数据集存储路径
            "dataset_path_dict": {},
            # 预处理数据存储路径
            "save_path": tmp_path + "/preprocess/",
            # Word2Vec向量大小
            "vector_size": {},
            # 列名映射
            "columns_map": {},
            # 分位点
            "quantile": {
                "start": 1,
                "end": 0
            },
            # 训练集、验证集、测试集划分比例 8:1:1（0.2代表20%的验证集、测试集，0.5代表在20%中的验证集训练集中按照50%比例再进行划分）
            "split_percentage": [0.2, 0.5],
        }

        # 日志配置
        self.LOG = {"file_path": tmp_path + "/log/"}

        # 模型配置
        self.MODEL = {
            # 模型名
            "name": "",
            # 超参
            "HyperParameter": {},
            # 存储路径
            "save_path": tmp_path + "/model/"
        }

        # 模型配置
        if model == "ProbS":
            self.MODEL["name"] = "ProbS"
            self.MODEL["HyperParameter"] = {
                # 混合比例常数
                "lambda": 1
            }
        elif model == "HeatS":
            self.MODEL["name"] = "HeatS"
            self.MODEL["HyperParameter"] = {
                # 混合比例常数
                "lambda": 0
            }
        elif model == "HybridS":
            self.MODEL["name"] = "HybridS"
            self.MODEL["HyperParameter"] = {
                # 混合比例常数
                "lambda": 0.6
            }
        elif model == "LightGCN":
            self.MODEL["name"] = "LightGCN"
            self.MODEL["HyperParameter"] = {
                # 种子值
                "seed": 42,
                # Embedding维度
                "embedding_dim": 64,
                # GCN层数
                "layers": 3,
                # 学习率
                "lr": 1e-3,
                # 学习率衰减因子
                "gamma": 0.95,
                # 训练轮次
                "epochs": 10000,
                # 每多少epoch评估一次模型
                "epoch_per_eval": 200,
                # 每多少epoch衰减一次学习率
                "epoch_per_lr_decay": 200,
                # Mini-batch批次大小
                "batch_size": 1024,
                # 控制BPR损失L2正则化强度
                "epsilon": 1e-6
            }
        elif model == "LightGCNOpti":
            self.MODEL["name"] = "LightGCNOpti"
            self.MODEL["HyperParameter"] = {
                # 种子值
                "seed": 42,
                # Embedding维度
                "embedding_dim": 64,
                # GCN层数
                "layers": 3,
                # 学习率
                "lr": 1e-3,
                # 学习率衰减因子
                "gamma": 0.95,
                # 训练轮次
                "epochs": 10000,
                # 每多少epoch评估一次模型
                "epoch_per_eval": 200,
                # 每多少epoch衰减一次学习率
                "epoch_per_lr_decay": 200,
                # Mini-batch批次大小
                "batch_size": 1024,
                # 控制BPR损失L2正则化强度
                "epsilon": 1e-6
            }
        elif model == "SpreadLightGCN":
            self.MODEL["name"] = "SpreadLightGCN"
            self.MODEL["HyperParameter"] = {
                # 种子值
                "seed": 42,
                # Embedding维度
                "embedding_dim": 64,
                # GCN层数
                "layers": 3,
                # 学习率
                "lr": 1e-3,
                # 学习率衰减因子
                "gamma": 0.95,
                # 训练轮次
                "epochs": 10000,
                # 每多少epoch评估一次模型
                "epoch_per_eval": 200,
                # 每多少epoch衰减一次学习率
                "epoch_per_lr_decay": 200,
                # Mini-batch批次大小
                "batch_size": 1024,
                # 控制BPR损失L2正则化强度
                "epsilon": 1e-6,
                # 混合比例常数
                "lambda": 0.85
            }
        elif model == "SpreadLightGCNOpti":
            self.MODEL["name"] = "SpreadLightGCNOpti"
            self.MODEL["HyperParameter"] = {
                # 种子值
                "seed": 42,
                # Embedding维度
                "embedding_dim": 64,
                # GCN层数
                "layers": 3,
                # 学习率
                "lr": 1e-3,
                # 学习率衰减因子
                "gamma": 0.95,
                # 训练轮次
                "epochs": 10000,
                # 每多少epoch评估一次模型
                "epoch_per_eval": 200,
                # 每多少epoch衰减一次学习率
                "epoch_per_lr_decay": 200,
                # Mini-batch批次大小
                "batch_size": 1024,
                # 控制BPR损失L2正则化强度
                "epsilon": 1e-6,
                # 混合比例常数
                "lambda": 0.6
            }

        # 评估配置
        self.EVALUATION = {
            # 存储路径
            "save_path": tmp_path + "/evaluation/"
        }

        # 推荐配置
        self.RECOMMEND = {
            # 推荐候选集大小
            "k": 100,
            # 存储路径
            "save_path": tmp_path + "/recommend/"
        }

        # 图片配置
        self.PICTURES = {
            # 存储路径
            "save_path": tmp_path + "/pictures/"
        }

        # MovieLens数据集预处理参数设置
        if dataset == "movielens":
            self.PREPROCESSING["dataset_path_dict"] = {
                "users": "/data/datasets/recommend/ml-100k/u.user",
                "items": "/data/datasets/recommend/ml-100k/u.item",
                "rating": "/data/datasets/recommend/ml-100k/u.data",
                "occupation": "/data/datasets/recommend/ml-100k/u.occupation"
            }
            self.PREPROCESSING["columns_map"] = {
                "user_id": "user",
                "item_id": "item",
                "rating": "rating",
                "rating_time": "timestamp"
            }
            self.PREPROCESSING["quantile"] = {
                "start": 1,
                "end": 0
            }
            self.PREPROCESSING["vector_size"] = {
                "title": 5,
                "content": 20
            }
        # douban数据集预处理参数设置
        elif dataset == "douban":
            self.PREPROCESSING["dataset_path_dict"] = {
                "users":"/data/datasets/recommend/douban/users.csv",
                "items":"/data/datasets/recommend/douban/movies.csv",
                "rating":"/data/datasets/recommend/douban/ratings.csv"
            }
            self.PREPROCESSING["columns_map"] = {
                "user_id": "USER_MD5",
                "item_id": "MOVIE_ID",
                "rating": "RATING",
                "rating_time": "RATING_TIME"
            }
            self.PREPROCESSING["quantile"] = {
                "start": 0.991,
                "end": 0.99
            }
            self.PREPROCESSING["vector_size"] = {
                "title": 3,
                "content": 20
            }

            self.RECOMMEND["target_user"] = "1a76e2591cd2f3740ccb7f198dace22a"

        


# MovieLens 数据集
dataset = "movielens"
# douban 数据集
# dataset = "douban"

# ProbS 推荐
# model = "ProbS"
# HeatS 推荐
# model = "HeatS"
# HybridS 推荐
# model = "HybridS"
# LightGCN 推荐
# model = "LightGCN"
# 嵌入优化的LightGCN推荐
# model = "LightGCNOpti"
# Spread+LightGCN 推荐
# model = "SpreadLightGCN"
# Spread+嵌入优化的LightGCN 推荐
model = "SpreadLightGCNOpti"


# 开发环境
env = "dev"
# 实验室服务器环境
# env = "prod"
cfg = Config(env, dataset, model).config
