# coding=UTF-8
"""
@Description   日志控制器  
@Author        Alex_McAvoy
@Date          2023-12-12 11:14:02
"""

import logging
import datetime

from const import cfg


class Logger:
    def __init__(self, save_path: str = None) -> None:
        """
        @description 构造函数
        @param {*} self 类实例化对象
        @param {str} save_path 日志存储路径
        """
        # 创建记录器
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.DEBUG)

        # 日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        # 设置级别
        console_handler.setLevel(logging.DEBUG)
        # 设置格式
        console_handler.setFormatter(formatter)
        # 将处理器添加到记录器中
        self.logger.addHandler(console_handler)

        # 写入到文件
        if save_path is not None:
            # 获取当前时间
            current_time = datetime.datetime.now()
            # 将当前时间转换为时间戳
            timestamp = int(current_time.timestamp())

            # 创建文件处理器
            file_handler = logging.FileHandler(save_path + "app_" + str(timestamp) + ".log", encoding="utf-8")
            # 设置级别
            file_handler.setLevel(logging.INFO)
            # 设置格式
            file_handler.setFormatter(formatter)
            # 将处理器添加到记录器中
            self.logger.addHandler(file_handler)

    def debug(self, message: str):
        """
        @description 调试信息
        @param {*} self 类实例化对象
        @param {str} message 信息
        """
        self.logger.debug(message)

    def info(self, message: str):
        """
        @description 信息
        @param {*} self 类实例化对象
        @param {str} message 信息
        """
        self.logger.info(message)

    def warning(self, message: str):
        """
        @description 警告
        @param {*} self 类实例化对象
        @param {str} message 信息
        """
        self.logger.warning(message)

    def error(self, message: str):
        """
        @description 错误
        @param {*} self 类实例化对象
        @param {str} message 信息
        """
        self.logger.error(message)

    def critical(self, message: str):
        """
        @description 严重错误
        @param {*} self 类实例化对象
        @param {str} message 信息
        """
        self.logger.critical(message)


# 日志控制器
logger = Logger(save_path=cfg.LOG["file_path"])
