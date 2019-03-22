# coding:utf-8
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import scale
from sklearn.naive_bayes import GaussianNB

VECTOR_DIR = 'D:/TBdata/baike26g_news13g_novel229g_128.bin'

MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
TEST_SPLIT = 0.2

print('(1) load texts...')
train_texts = open('D:/TBdata/我的语料库/random/01/traintext-ran3.txt', encoding='utf-8').read().split('\n')
train_labels = open('D:/TBdata/我的语料库/random/01/trainlabel-ran3.txt', encoding='utf-8' ).read().split('\n')
test_texts = open('D:/TBdata/我的语料库/random/01/testtext-ran3.txt', encoding='utf-8').read().split('\n')
test_labels = open('D:/TBdata/我的语料库/random/01/testlabel-ran3.txt', encoding='utf-8').read().split('\n')
all_texts = train_texts + test_texts
all_labels = train_labels + test_labels


print('(2) doc to var...')
import gensim
import numpy as np

w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True)


def buildWordVector(text, size):
    '''
        利用函数获得每个文本中所有词向量的平均值来表征该特征向量。
    '''
    vec = np.zeros(128).reshape((1, size))
    count = 0
    for word in text:
        try:
            vec += w2v_model[word].reshape((1, 128))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


'''获取需要所有文档的词向量，并且标准化出来'''
x_train1 = np.concatenate([buildWordVector(x, 128) for x in train_texts])
print("the shape of train is " + repr(x_train1.shape))
x_train = scale(x_train1)
x_test1 = np.concatenate([buildWordVector(x, 128) for x in test_texts])
print("the shape of train is " + repr(x_test1.shape))
x_test = scale(x_test1)
y_train = train_labels
y_test = test_labels

clf = GaussianNB()
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
num = 0
preds = preds.tolist()
for i, pred in enumerate(preds):
    if int(pred) == int(y_test[i]):
        num += 1
print('precision_score:' + str(float(num) / len(preds)))

