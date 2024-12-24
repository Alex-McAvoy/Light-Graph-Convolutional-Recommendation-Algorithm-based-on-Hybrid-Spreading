#!/usr/bin/env python
# coding=utf-8
"""
@Description   通用绘图函数
@Author        Alex_McAvoy
@Date          2024-10-18 11:20:49
"""

import matplotlib.pyplot as plt

def plotMetric(xpoints: list, ypoints: list, xlabel: str, ylabel: str, title: str, save_path: str) -> None:
    """
    @description 折线图绘制工具
    @param {list} xpoints X轴数据点
    @param {list} ypoints Y轴数据点
    @param {str} lable 线条标签，用于创建图例
    @param {str} xlabel X轴标签
    @param {str} ylabel Y轴标签
    @param {str} title 题目
    @param {str} save_path 存储路径
    """
    plt.figure()
    plt.plot(xpoints, ypoints)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path)
