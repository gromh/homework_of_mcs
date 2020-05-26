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
