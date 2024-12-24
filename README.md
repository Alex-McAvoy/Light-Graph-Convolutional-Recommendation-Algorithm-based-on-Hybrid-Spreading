# Light-Graph-Convolutional-Recommendation-Algorithm-based-on-Hybrid-Spreading
基于混合扩散的轻量级图卷积推荐算法（Light Graph Convolutional Recommendation Algorithm based on Hybrid Spreading，LGCNHS）

## 项目运行

1. 克隆项目

```sh
git clone https://github.com/Alex-McAvoy/Legends-of-the-Three-Kingdoms-Offline-Assistant.git
```

2. 进入项目文件夹并安装conda环境

```sh
cd ./Legends-of-the-Three-Kingdoms-Offline-Assistant
npm i
```

3. 进入项目文件夹并安装conda环境

```sh
conda env create -f environment.yml -n RS
conda activate RS
```

3. 修改配置文件 `const.py`
4. 运行推荐函数

```sh
python3 main.py
```

## 文件目录

```
filetree 
├── /draw/
│  ├── ablation.ipynb
│  └── findLambda.ipynb
├── /metrtics/
│  ├── accurate.py
│  └── diversity.py
├── /model/
│  ├── /LightGCN/
│  │  ├── evaluation.py
│  │  ├── loss.py
│  │  ├── model.py
│  │  ├── recommend.py
│  │  └── train.py
│  ├── /LightGCNOpti/
│  │  ├── evaluation.py
│  │  ├── loss.py
│  │  ├── model.py
│  │  ├── recommend.py
│  │  └── train.py
│  ├── /SpreadLightGCN/
│  │  ├── model.py
│  │  └── recommend.py
│  ├── /SpreadLightGCNOpti/
│  │  ├── model.py
│  │  └── recommend.py
│  └── /SpreadMethod/
│     ├── model.py
│     └── recommend.py
├── /processing/
│  ├── handleData.py
│  ├── handleDouban.py
│  ├── handleFeature.py
│  └── handleMovielens.py
├── /utils/
│  ├── graph.py
│  ├── log.py
│  ├── picture.py
│  ├── trans.py
│  └── wrapper.py
├── /waste/
├── const.py
├── evaluationMetrics.py
├── findLambda.py
├── main.py
└── README.md
```
