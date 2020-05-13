'''
Think1:
    常用的文本分类方法有TF-IDF模型，决策树， 贝叶斯模型， SVM模型， 神经网络分类， KNN分类
    深度学习 fasttext模型，TextCNN模型，TextRNN模型， TextRNN + Attention模型
Think2：
    因为每个j对j-1状态的偏导都含有tanh函数的偏导，tanh函数偏导最大为1而且经常趋于0，就会导致梯度消失。
    如果更换tanh函数，比如relu，梯度中会存在很多w的连乘，会产生梯度爆炸问题。
    LSTM引入遗忘门、输入门和输出门让不同的序列相加而不是相乘，解决上述梯度消失和爆炸问题。
'''


import numpy as np
import tensorflow.keras as kr


# 读取词汇表
def read_vocab(vocab_dir):
    with open(vocab_dir, 'r', encoding='utf-8', errors='ignore') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


# 读取分类目录，固定
def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categories = [x for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


# 将文件转换为id表示
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])  # 将每句话id化
        label_id.append(cat_to_id[labels[i]])  # 每句话对应的类别的id
    #
    # # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    #
    return x_pad, y_pad


# 获取文本的类别及其对应id的字典
categories, cat_to_id = read_category()
print(categories)
# 获取训练文本中所有出现过的字及其所对应的id
words, word_to_id = read_vocab('cnews.vocab.txt')

# print(words)
# print(word_to_id)
# print(word_to_id)
# 获取字数
vocab_size = len(words)

# 数据加载及分批
# 获取训练数据每个字的id和对应标签的one-hot形式
x_train, y_train = process_file('cnews.train.txt', word_to_id, cat_to_id, 600)
print('x_train=', x_train)
x_val, y_val = process_file('cnews.val.txt', word_to_id, cat_to_id, 600)

# 搭建模型

from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, ReLU

# 参数设置
lr = 0.001
batch_size = 128

model = Sequential()

model.add(Embedding(50000, 128))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(ReLU())
model.add(GRU(128))
model.add(Dropout(0.5))
model.add(ReLU())
model.add(Dense(10, activation='softmax'))
# 优化器
opt = Adam(lr=lr)

# 模型搭建
model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])

# 模型训练
model.fit(x_train, y_train, batch_size, 10,
          validation_data=[x_val, y_val])

#50000/50000 [==============================] - 890s 18ms/step - loss: 0.0637 - accuracy: 0.9837 - val_loss: 0.3456 - val_accuracy: 0.9196