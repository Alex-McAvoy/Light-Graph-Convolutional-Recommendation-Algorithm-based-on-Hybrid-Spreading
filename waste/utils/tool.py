"""
@Description   基本工具
@Author        Alex_McAvoy
@Date          2024-02-04 22:23:03
"""

import os


def isFolderEmpty(folder_path: str):
    """
    @description 判断文件夹是否为空
    @param {str} folder_path 文件夹路径
    @return {bool} 
        - True：为空
        - False：不为空
    """
    # 获取folder_path内的所有文件和文件夹列表
    files_in_folder = os.listdir(folder_path)
    # 判空
    if len(files_in_folder) == 0:
        return True
    else:
        return False