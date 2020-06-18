from gensim.models import word2vec

word_file = './three_kingdoms/segment/seg_threekingdoms.txt'
senstence = word2vec.PathLineSentences(word_file)

model1 = word2vec.Word2Vec(senstence, size=128, window=3, min_count=2)

print(model1.wv.most_similar('曹操'))
print(model1.wv.most_similar(positive=['曹操', '刘备'], negative=['张飞']))
model1.save('./models/word2Vec_threekingdim.model')

'''
[('孙权', 0.9883049726486206), ('先主', 0.9877791404724121), ('回报', 0.9873332977294922), 
('夫人', 0.9860264658927917), ('关公', 0.9857215881347656), ('孔明', 0.9843080043792725), 
('荆州', 0.983728289604187), ('周瑜', 0.9833334684371948), ('往', 0.9825193285942078), ('又', 0.9818975329399109)]

[('丞相', 0.9887984395027161), ('臣', 0.9875719547271729), ('某', 0.9866517782211304), 
('此', 0.9865485429763794), ('大叫', 0.9859899282455444), ('皆曰', 0.9858393669128418), 
('朕', 0.9830409288406372), ('书略', 0.9822883605957031), ('乃曰', 0.9815787076950073), ('既', 0.9811386466026306)]
'''