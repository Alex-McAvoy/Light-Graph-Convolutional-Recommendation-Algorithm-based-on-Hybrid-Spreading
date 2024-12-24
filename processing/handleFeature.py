#!/usr/bin/env python
# coding=utf-8
"""
@Description   特征处理通用函数
@Author        Alex_McAvoy
@Date          2024-09-26 14:47:09
"""

import re
import numpy as np

import jieba
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

def ageMap(age: int) -> int:
    """
    @description 年龄映射
    @param {int} age 年龄
    @return {int} 映射值
    """
    if age >= 1 and age <= 7:
        return 1
    if age >= 8 and age <= 16:
        return 2
    if age >= 17 and age <= 29:
        return 3
    if age >= 30 and age <= 39:
        return 4
    if age >= 40 and age <= 49:
        return 5
    if age >= 50 and age <= 59:
        return 6
    if age >= 60:
        return 7


def yearMap(year: str) -> int:
    """
    @description 年份映射
    @param {str} year 年份
    @return {int} 映射值
    """
    if year == "nan":
        return 0
    year = int(year)
    if year < 1970:
        return 1
    elif 1970 <= year < 1980:
        return 2
    elif 1980 <= year < 1990:
        return 3
    elif 1990 <= year < 2000:
        return 4
    elif 2000 <= year < 2010:
        return 5
    else:
        return 6


def genreCleanMap(rows: list) -> list:
    """
    @description 类别清洗
    @param {list} rows 类别列表
    @return {list} 类别清洗列表
    """
    # 类别清洗字典
    genre_replace_dict = {
        "動畫": "动画",
        "Animation": "动画",
        "音樂": "音乐",
        "Music": "音乐",
        "動作": "动作",
        "Action": "动作",
        "兒童": "儿童",
        "Kids": "儿童",
        "紀錄片": "纪录片",
        "Documentary": "纪录片",
        "歷史": "历史",
        "History": "历史",
        "喜劇": "喜剧",
        "Comedy": "喜剧",
        "懸疑": "悬疑",
        "Mystery": "悬疑",
        "傳記": "传记",
        "Biography": "传记",
        "News": "传记",
        "愛情": "爱情",
        "Romance": "爱情",
        "驚悚": "惊悚",
        "Thriller": "惊悚",
        "惊栗": "惊悚",
        "劇情": "剧情",
        "Talk-Show": "脱口秀",
        "Reality-TV": "真人秀",
        "Drama": "戏曲",
        "Adult": "成人",
    }
    return [genre_replace_dict.get(row, row) for row in rows]


def languageMap(rows: list) -> list:
    """
    @description 语言映射
    @param {list} rows 语言列表
    @return {list} 语言映射列表
    """

    if len(rows) == 0:
        return [0]

    ls = []
    for row in rows:
        if row == "汉语普通话":
            ls.append(1)
        elif row == "英语":
            ls.append(2)
        else:
            ls.append(3)

    return list(set(ls))


def regionMap(rows: list) -> list:
    """
    @description 区域映射
    @param {list} rows 区域列表
    @return {list} 区域映射列表
    """

    if len(rows) == 0:
        return [0]

    ls = []
    for row in rows:
        if row == "中国大陆":
            ls.append(1)
        elif row == "美国":
            ls.append(2)
        else:
            ls.append(3)

    return list(set(ls))


def durationMap(duration: float) -> int:
    """
    @description 时长映射
    @param {float} duration 时长
    @return {int} 映射值
    """
    if duration >= 0 and duration <= 30:
        return 1
    if duration > 30 and duration <= 60:
        return 2
    if duration > 60 and duration <= 90:
        return 3
    if duration > 90 and duration <= 120:
        return 4
    if duration > 120 and duration <= 150:
        return 5
    if duration > 150:
        return 6


def preprocessText(text: str) -> str:
    """
    @description 文本数据预处理
    @param {str} text 文本
    @return {str} words 处理后的分词列表
    """

    # 保证是str类型数据
    text = str(text)

    # 去除标点符号
    text = re.sub(r"[^\w\s]", "", text)

    # 去除数字
    text = re.sub(r"\d+", "", text)

    # 转换为小写
    text = text.lower()

    # 分词
    words = jieba.lcut(text)
    words = [elem for elem in words if elem.strip() != ""]

    # 初始化词形还原器
    lemmatizer = WordNetLemmatizer()

    # 进行词形还原
    words = [lemmatizer.lemmatize(word) for word in words]

    # 去除停用词
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # 连接各单词
    # words = " ".join(words)

    return words


def getWord2Vec(word_lists: list, vector_size: int) -> np.ndarray:
    """
    @description 获取Word2Vec向量
    @param {list} word_lists 单词列表
    @param {int} vector_size Word2Vec向量长度
    @return {np} 所有单词列表的Word2Vec向量
    """
    # 建立Word2Vec模型
    model = Word2Vec(
        word_lists,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
    )

    vectors = []
    # 枚举所有单词列表
    for word_list in word_lists:
        if len(word_list) == 0:
            # 若当前单词列表为空，设为0向量
            vectors.append(np.zeros(vector_size))
        else:
            vector = []
            # 枚举当前单词列表的所有单词，计算特征向量
            for word in word_list:
                vector.append(model.wv[word])
            # 当前单词列表的特征向量为所有单词的特征向量的平均
            vectors.append(np.mean(vector, axis=0))

    # 垂直方向堆叠各单词列表的Word2Vec向量
    vectors = np.vstack(vectors)
    return vectors
