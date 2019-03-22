#coding:utf-8


from sklearn.preprocessing import scale
from itertools import cycle

VECTOR_DIR = 'D:/TBdata/baike26g_news13g_novel229g_128.bin'

MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
TEST_SPLIT = 0.2


print ('(1) load texts...')
train_texts = open('D:/TBdata/我的语料库/random/01/traintext-ran3.txt', encoding='utf-8').read().split('\n')
train_labels = open('D:/TBdata/我的语料库/random/01/trainlabel-ran3.txt', encoding='utf-8' ).read().split('\n')
test_texts = open('D:/TBdata/我的语料库/random/01/testtext-ran3.txt', encoding='utf-8').read().split('\n')
test_labels = open('D:/TBdata/我的语料库/random/01/testlabel-ran3.txt', encoding='utf-8').read().split('\n')

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
x_train = scale(x_train1)
x_test1 = np.concatenate([buildWordVector(x, 128) for x in test_texts])
x_test = scale(x_test1)
y_train = train_labels
y_test = test_labels

print('(3) SVM...')
from sklearn.svm import SVC
from sklearn.metrics import classification_report

svclf = SVC(kernel='sigmoid')
svclf.fit(x_train, y_train)
preds = svclf.predict(x_test)
num = 0
preds = preds.tolist()
for i, pred in enumerate(preds):
    if int(pred) == int(y_test[i]):
        num += 1
print('precision_score:' + str(float(num) / len(preds)))

print("classification_report(left: labels):")
print(classification_report(y_test, preds))

'''
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from scipy import interp

y_score = preds
lw = 2
n_classes = 2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area


# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
'''




        




