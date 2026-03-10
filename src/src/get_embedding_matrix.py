from gensim.models import FastText

text = [
"锂金属电池，简单来说是一种以金属锂作为负极的下一代高能量电池，你可以把它和我们目前最常用的锂离子电池做个对比，就很好理解了。",
"锂金属电池和锂离子电池都属于锂电池家族。它们最大的不同在负极材料，锂离子电池使用石墨做负极，而锂金属电池则使用纯金属锂。",
"用金属锂做负极，能极大提升电池负极储存锂离子的能力，从而显著提高电池的能量密度。简单讲，就是在同样重量或体积下，能装更多的电，让设备续航更久。理论上，金属锂能让电动汽车的续航里程直接翻倍。",
"虽然性能诱人，但锂金属电池也一直面临难题。金属锂性质非常活泼，容易与电解质发生不良反应，带来安全性风险。",
"而且在充放电过程中，容易长出锂枝晶（像树枝一样的金属锂），可能刺穿电池内部隔膜导致短路，严重影响电池的循环寿命。"
]
import jieba
jieba_cut_list = []
for line in text:
    jieba_cut = list(jieba.cut(line))
    jieba_cut_list.append(jieba_cut)

model = FastText(vector_size=4, window=3, min_count=1, sentences=jieba_cut_list, epochs=10)

model.build_vocab(jieba_cut_list)
model.train(jieba_cut_list, total_examples=model.corpus_count, epochs=10)#这里使用笔者给出的固定格式即可
model.save("./xiaohua_fasttext_model_jieba.model")

model = FastText.load("./xiaohua_fasttext_model_jieba.model")

print(model.wv.key_to_index)
print(model.wv.index_to_key)
print(model.wv.vectors[:3])
print(len(model.wv.vectors))
print(len(model.wv.index_to_key))

embedding = (model.wv["卷积","神经网络"])
print(embedding)
