# FastText 学习项目

本项目使用 gensim 库训练一个简单的 FastText 词向量模型，语料为关于锂金属电池的中文文本，并用 jieba 进行分词。

## 功能

- 加载中文文本
- 使用 jieba 分词
- 训练 FastText 词向量
- 保存/加载模型
- 查询词向量

## 环境配置

```bash
pip install -r requirements.txt
```

## JAX
利用 JAX 强大的自动微分和即时编译（JIT）功能，构建一个以词向量为基础的分类器。将 FastText 生成的向量作为 embeddings 层的初始化权重，然后在此基础上训练一个简单的分类模型。
