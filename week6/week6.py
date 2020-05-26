'''
Thinking1 ： 什么是监督学习，无监督学习，半监督学习？
  监督学习是有标签学习如线性回归，逻辑回归， SVM，决策树，朴素贝叶斯等
  无监督学习是无标签学习有KMEANS，PCA等
  半监督学习，是少量数据有标签，利用没有标签的数据来学习整个数据的结果
Thinking2 ： K-means中的k值如何选取？
    K越大分类越精细，但是可能误差很大。可以利用手肘法计算不同的K将损失函数找到拐点K值，也可以用Gap Statistic方法找到Gap（K）最大的K值
Thinking3：bagging指什么?
    将若干的弱分类器投票选出一个强分类器，用于解决过拟合问题，减少方差
    boosting用于解决欠拟合
Thinking4： 表征学习和半监督学习的区别？
    表征学习是在数据中找到新特征，对数据进行精简
    半监督学习是为了拓展数据，提高无标签数据的可利用性，增加数据
'''



import os
import pickle
from collections import defaultdict

import editdistance
import jieba
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer

# stop words
with open('chinese_stopwords.txt', 'r', encoding='utf-8') as file:
    stopwords = [i[:-1] for i in file.readlines()]

# download data
news = pd.read_csv('sqlResult.csv', encoding='gb18030')
# print(news.shape)
# print(news.head())

# print(news[news.content.isna()].head())

news = news.dropna(subset=['content'])
print(news.shape)


def split_text(text):
    text = text.replace(' ', '')
    text = text.replace('\n', '')
    text2 = jieba.cut(text.strip())
    result = ' '.join([w for w in text2 if w not in stopwords])
    return result


# import pickle, os
if not os.path.exists('corpus.pkl'):
    corpus = list(map(split_text, [str(i) for i in news.content]))
    print(corpus[0])
    print(len(corpus))
    print(corpus[1])
    with open('corpus.pkl', 'wb') as file:
        pickle.dump(corpus, file)
else:
    with open('corpus.pkl', 'rb') as file:
        corpus = pickle.load(file)

# 计算TF-IDF
countvectorizer = CountVectorizer(encoding='gb18030', min_df=0.015)
tfidtransformer = TfidfTransformer()
countvector = countvectorizer.fit_transform(corpus)
tfidf = tfidtransformer.fit_transform(countvector)
# print(tfidf.shape)

label = list(map(lambda source: 1 if '新华' in str(source) else 0, news.source))

X_train, X_test, y_train, y_test = train_test_split(tfidf.toarray(), label, test_size=0.3)
clf = MultinomialNB()
clf.fit(X_train, y_train)
prediction = clf.predict(tfidf.toarray())
labels = np.array(label)

compare_news_index = pd.DataFrame({'prediction': prediction, 'labels': labels})
# 计算所有可以文章的index
copy_news_index = compare_news_index[(compare_news_index['prediction'] == 1) & (compare_news_index['labels'] != 1)]
# 计算所有新华社文章的index
xinhuashe_news_index = compare_news_index[(compare_news_index['labels'] == 1)].index
# print(len(copy_news_index))
# 归一化
normalizer = Normalizer()
scaled_array = normalizer.fit_transform(tfidf.toarray())

# kmeans
if not os.path.exists('label.pkl'):
    kmeans = KMeans(n_clusters=25)
    k_labels = kmeans.fit_predict(scaled_array)
    with open('label.pkl', 'wb') as file:
        pickle.dump(k_labels, file)
    print(k_labels.shape)
else:
    with open('label.pkl', 'rb') as file:
        k_labels = pickle.load(file)

# 创建id_class
if not os.path.exists('id_class.pkl'):
    id_class = {index: class_ for index, class_ in enumerate(k_labels)}
    with open('id_class.pkl', 'wb') as file:
        pickle.dump(id_class, file)
else:
    with open('id_class.pkl', 'rb') as file:
        id_class = pickle.load(file)

if not os.path.exists('class_id.pkl'):
    class_id = defaultdict(set)
    for index, class_ in id_class.items():
        if index in xinhuashe_news_index.tolist():
            class_id[class_].add(index)
    with open('class_id.pkl', 'wb') as file:
        pickle.dump(class_id, file)
else:
    with open('class_id.pkl', 'rb') as file:
        class_id = pickle.load(file)


# 找到相似文章
def find_similar_text(cpindex, top=10):
    dist_dict = {i: cosine_similarity(tfidf[cpindex], tfidf[i]) for i in class_id[id_class[cpindex]]}
    return sorted(dist_dict.items(), key=lambda x: x[1][0], reverse=True)[:top]


cpindex = 2000
similar_list = find_similar_text(cpindex)
# print(similar_list)
print('怀疑抄袭\n', news.iloc[cpindex].content)
# 相似原文
similar2 = similar_list[0][0]
print('相似原文', news.iloc[similar2].content)

# 文本编辑距离
mean_lengh_news = (len(news.iloc[similar2].content) + len(news.iloc[cpindex].content)) // 2
print('编辑距离', editdistance.eval(corpus[cpindex], corpus[similar2]) / mean_lengh_news)
