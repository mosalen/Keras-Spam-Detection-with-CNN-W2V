#coding:utf-8
import sys
import keras

VECTOR_DIR = 'D:/TBdata/baike26g_news13g_novel229g_128.bin'
#'D:/first/text_classification/wiki.zh.vector.bin'


MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2


print ('(1) load texts...')
train_texts = open('D:/TBdata/我的语料库/random/01/traintext-ran3.txt', encoding='utf-8').read().split('\n')
train_labels = open('D:/TBdata/我的语料库/random/01/trainlabel-ran3.txt', encoding='utf-8' ).read().split('\n')
test_texts = open('D:/TBdata/我的语料库/random/01/testtext-ran3.txt', encoding='utf-8').read().split('\n')
test_labels = open('D:/TBdata/我的语料库/random/01/testlabel-ran3.txt', encoding='utf-8').read().split('\n')
all_texts = train_texts + test_texts
all_labels = train_labels + test_labels


print ('(2) doc to var...')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)
sequences = tokenizer.texts_to_sequences(all_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(all_labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


print ('(3) split data set...')
# split the data into training set, validation set, and test set
p1 = int(len(data)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(data)*(1-TEST_SPLIT))
x_train = data[:p1]
y_train = labels[:p1]
x_val = data[p1:p2]
y_val = labels[p1:p2]
x_test = data[p2:]
y_test = labels[p2:]
print ('train docs: '+str(len(x_train)))
print ('val docs: '+str(len(x_val)))
print ('test docs: '+str(len(x_test)))


print ('(4) load word2vec as embedding...')
import gensim
from keras.utils import plot_model
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True)
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
not_in_model = 0
in_model = 0
for word, i in word_index.items(): 
    if word in w2v_model:
        in_model += 1
        embedding_matrix[i] = np.asarray(w2v_model[word], dtype='float32')
    else:
        not_in_model += 1
print (str(not_in_model)+' words not in w2v model')
from keras.layers import Embedding
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


print ('(5) training model...')
from keras.layers import Dense, Input, Flatten, Dropout, LSTM
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D
from keras.models import Sequential


model = Sequential()
model.add(embedding_layer)
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.2))
model.add(Dense(labels.shape[1], activation='sigmoid'))
model.summary()
plot_model(model, to_file='D:\TBdata\验证结果\model-lstm.png',show_shapes=True)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
print (model.metrics_names)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=64)
model.save('word_vector_cnn.h5')

print ('(6) testing model...')
print (model.evaluate(x_test, y_test))

'''
predict_text = open('D:/TBdata/我的语料库/random/predict-text.txt', encoding='utf-8').read().split('\n')
predict_label = open('D:/TBdata/我的语料库/random/predict-label.txt', encoding='utf-8').read().split('\n')


prediction = model.predict(x_pre)
print("模型预测结果", prediction)

import csv

print("Saving evaluation")
prediction_human_readable = np.column_stack((np.array(x_pre), prediction))
with open("D:/TBdata/验证结果/Prediction.csv", 'w') as f:
    csv.writer(f).writerows(prediction_human_readable)

'''

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

y_score = model.predict(x_test)
y_pred_labels = np.argmax(y_score, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

lw = 2
n_classes = 2


print("computing f1 score...")
cm1 = confusion_matrix(y_test_labels, y_pred_labels)
print(cm1)
TPR = np.diag(cm1)
FPR = []
for i in range(n_classes):
    FPR.append(sum(cm1[i, :]) - cm1[i, i])
FNR = []
for i in range(n_classes):
    FNR.append(sum(cm1[i, :]) - cm1[i, i])
TNR = []
for i in range(n_classes):
    temp = np.delete(cm1, i, 0)   # delete ith row
    temp = np.delete(temp, i, 1)  # delete ith column
    TNR.append(sum(sum(temp)))
l = len(y_test)
for i in range(n_classes):
    print(TPR[i] + FPR[i] + FNR[i] + TNR[i] == l)

precision = TPR / (TPR + FPR)
print(precision)
recall = TPR / (TPR + FNR)
print(recall)
f1_score = 2.0 * precision * recall / (precision + recall)
print(f1_score)

print("classification_report(left: labels):")
print(classification_report(y_test_labels, y_pred_labels))
