import jax
import jax.numpy as jnp
import numpy as np
from gensim.models import FastText
import jieba

# 1. 原始数据与标签
texts = [
    "锂金属电池，简单来说是一种以金属锂作为负极的下一代高能量电池，你可以把它和我们目前最常用的锂离子电池做个对比，就很好理解了。",
    "锂金属电池和锂离子电池都属于锂电池家族。它们最大的不同在负极材料，锂离子电池使用石墨做负极，而锂金属电池则使用纯金属锂。",
    "用金属锂做负极，能极大提升电池负极储存锂离子的能力，从而显著提高电池的能量密度。简单讲，就是在同样重量或体积下，能装更多的电，让设备续航更久。理论上，金属锂能让电动汽车的续航里程直接翻倍。",
    "虽然性能诱人，但锂金属电池也一直面临难题。金属锂性质非常活泼，容易与电解质发生不良反应，带来安全性风险。",
    "而且在充放电过程中，容易长出锂枝晶（像树枝一样的金属锂），可能刺穿电池内部隔膜导致短路，严重影响电池的循环寿命。"
]
labels = [0, 0, 0, 1, 1]  # 0表示原理或者优点，1表示缺点

# 2. 分词并构建词汇表
tokenized_texts = [list(jieba.cut(t)) for t in texts]
vocab = {word for tokens in tokenized_texts for word in tokens}
word_to_idx = {word: i+1 for i, word in enumerate(vocab)} # 索引从1开始，0留给PAD
vocab_size = len(word_to_idx) + 1

# 3. 将文本转换为索引序列，并进行统一长度的填充（Padding）
max_len = max(len(tokens) for tokens in tokenized_texts)
def encode_and_pad(tokens):
    indices = [word_to_idx[word] for word in tokens]
    # 在序列前面填充0，使其长度一致。填充方式有多种，这里选择前填充。
    pad_len = max_len - len(indices)
    return [0] * pad_len + indices

X = jnp.array([encode_and_pad(tokens) for tokens in tokenized_texts])
y = jnp.array(labels)
# 4. 加载之前训练的 FastText 模型，并构建嵌入矩阵
fasttext_model = FastText.load("./fasttext_model.bin")
embedding_dim = fasttext_model.vector_size

# 初始化一个随机矩阵，作为嵌入层权重
embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim)).astype(np.float32)

# 将FastText中的词向量填充到矩阵的对应位置
for word, idx in word_to_idx.items():
    if word in fasttext_model.wv:
        embedding_matrix[idx] = fasttext_model.wv[word]
    # 如果词不在FastText词汇表中，则保持随机初始化

# 将numpy数组转换为JAX数组
embedding_matrix = jnp.array(embedding_matrix)
from jax import nn


def init_params(vocab_size, embed_dim, hidden_dim, num_classes, key):
    """初始化模型参数"""
    k1, k2, k3 = jax.random.split(key, 3)
    # 嵌入层权重直接用我们之前构建的矩阵，不需要随机初始化
    # 这里我们创建一个参数结构，包含嵌入矩阵和两个全连接层的权重与偏置
    params = {
        'embedding': embedding_matrix,  # 使用预训练好的矩阵
        'fc1': {
            'w': jax.random.normal(k1, (embed_dim, hidden_dim)) * 0.1,
            'b': jnp.zeros(hidden_dim)
        },
        'fc2': {
            'w': jax.random.normal(k2, (hidden_dim, num_classes)) * 0.1,
            'b': jnp.zeros(num_classes)
        }
    }
    return params


def forward(params, x):
    """前向传播函数"""
    # 1. 嵌入层查找
    # x 的形状: (batch_size, seq_len)
    # 嵌入后形状: (batch_size, seq_len, embed_dim)
    embedded = params['embedding'][x, :]

    # 2. 简单的池化操作：对序列维度求平均，得到整个句子的向量表示
    # 需要处理填充的0，避免它们影响平均值
    # 创建一个mask，标记非填充位置
    mask = (x > 0).astype(jnp.float32)[:, :, jnp.newaxis]
    sum_embedded = jnp.sum(embedded * mask, axis=1)
    token_counts = jnp.sum(mask, axis=1)
    # 避免除以0，对于长度为0的序列（理论上不会），直接设为0
    sentence_vector = jnp.where(token_counts > 0, sum_embedded / token_counts, 0)

    # 3. 第一个全连接层 + ReLU激活
    h = nn.relu(sentence_vector @ params['fc1']['w'] + params['fc1']['b'])

    # 4. 输出层 (logits)
    logits = h @ params['fc2']['w'] + params['fc2']['b']
    return logits
from jax import grad, jit

def loss_fn(params, x, y):
    """计算损失 (交叉熵)"""
    logits = forward(params, x)
    # 使用softmax交叉熵损失
    one_hot_y = jax.nn.one_hot(y, num_classes=2)
    loss = -jnp.mean(jnp.sum(one_hot_y * nn.log_softmax(logits), axis=-1))
    return loss

@jit
def update(params, x, y, lr=0.01):
    """单步更新参数"""
    grads = grad(loss_fn)(params, x, y)
    # 手动更新参数（演示用，实际可用optax优化器）
    new_params = {
        'embedding': params['embedding'], # 通常我们冻结预训练的嵌入层，或者用很小的学习率更新
        'fc1': {
            'w': params['fc1']['w'] - lr * grads['fc1']['w'],
            'b': params['fc1']['b'] - lr * grads['fc1']['b']
        },
        'fc2': {
            'w': params['fc2']['w'] - lr * grads['fc2']['w'],
            'b': params['fc2']['b'] - lr * grads['fc2']['b']
        }
    }
    return new_params
# 初始化参数
key = jax.random.PRNGKey(42)
params = init_params(vocab_size, embedding_dim, hidden_dim=8, num_classes=2, key=key)

# 训练循环
epochs = 100
for epoch in range(epochs):
    params = update(params, X, y, lr=0.05)
    if epoch % 20 == 0:
        loss = loss_fn(params, X, y)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# 简单预测
logits = forward(params, X)
predictions = jnp.argmax(logits, axis=-1)
print("\n最终预测结果:", predictions)
print("真实标签:       ", y)
