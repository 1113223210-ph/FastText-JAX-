from gensim.models import FastText
import jieba

# 原始文本
text = [
    "锂金属电池，简单来说是一种以金属锂作为负极的下一代高能量电池，你可以把它和我们目前最常用的锂离子电池做个对比，就很好理解了。",
    "锂金属电池和锂离子电池都属于锂电池家族。它们最大的不同在负极材料，锂离子电池使用石墨做负极，而锂金属电池则使用纯金属锂。",
    "用金属锂做负极，能极大提升电池负极储存锂离子的能力，从而显著提高电池的能量密度。简单讲，就是在同样重量或体积下，能装更多的电，让设备续航更久。理论上，金属锂能让电动汽车的续航里程直接翻倍。",
    "虽然性能诱人，但锂金属电池也一直面临难题。金属锂性质非常活泼，容易与电解质发生不良反应，带来安全性风险。",
    "而且在充放电过程中，容易长出锂枝晶（像树枝一样的金属锂），可能刺穿电池内部隔膜导致短路，严重影响电池的循环寿命。"
]

# 分词
sentences = [list(jieba.cut(line)) for line in text]

# 训练 FastText 模型
model = FastText(
    sentences=sentences,           # 直接传入分词后的句子
    vector_size=4,                  # 词向量维度
    window=3,                        # 上下文窗口大小
    min_count=1,                     # 忽略词频小于1的词
    epochs=10                         # 训练轮数
)

# 保存模型
model.save("./fasttext_model.bin")

# 加载模型
loaded_model = FastText.load("./fasttext_model.bin")

# 查看词汇表
print("词汇表:", loaded_model.wv.index_to_key)
print("词向量形状:", loaded_model.wv.vectors.shape)

# 查询多个词的词向量
words = ["锂金属", "枝晶"]  
for word in words:
    if word in loaded_model.wv:
        print(f"{word} 的词向量: {loaded_model.wv[word]}")
    else:
        print(f"词 '{word}' 不在词汇表中")
