from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import operator

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals




digits = load_digits()
print(digits.data.shape)
print(type(digits.data))
print("#############################")


# data = np.loadtxt(open("testcsv1.csv","rb"),delimiter=",",skiprows=0)
# print(type(data))


# #UTF-8编码格式csv文件数据读取
# df = pd.read_csv('testcsv1.csv') #返回一个DataFrame的对象，这个是pandas的一个数据结构
# df.columns=["Col1","Col2","Col3","Col4"]
 
# X = df[["Col1","Col2","Col3","Col4"]] #抽取前七列作为训练数据的各属性值
# X = np.array(X)
# X_ = X[:,1:len(X[0])]

# # print X
# print(type(X_))
# print(X_.shape)

print("*****************************")

print("*****************************")


X=np.loadtxt('datingTestSet2.txt',delimiter='\t')

X__=X[:,0:3]
print(type(X__))
X_, ranges, minVals = autoNorm(X__)
print(X_)
print("###########################")
print(X_.shape)
print("########################")
Y=X[:,3]
print(Y.shape)






X_tsne = TSNE(n_components=2,random_state=33).fit_transform(X_)
X_pca = PCA(n_components=2).fit_transform(X_)

ckpt_dir="images"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

plt.figure(figsize=(10, 5))

print("#################")
print(X_tsne.shape)
print(X_tsne[:,0].shape)
# print(X_tsne[:,1])
print(Y.shape)

plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=Y,label="t-SNE")
plt.legend()
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1],c=Y,label="PCA")
plt.legend()
plt.savefig('images/digits_tsne-pca.png', dpi=120)
plt.show()