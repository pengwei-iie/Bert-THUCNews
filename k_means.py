from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import cluster
import numpy as np

# 生成10*3的矩阵
data = []
with open('test.txt', 'r') as f:
    for line in f:
        line = line.strip().split()
        line = np.asarray(line, dtype=np.int)
        data.append(line)
print(data)
# 聚类为4类
estimator = KMeans(n_clusters=3)
# fit_predict表示拟合+预测，也可以分开写
res = estimator.fit_predict(data)
# 预测类别标签结果
lable_pred = estimator.labels_
# 各个类别的聚类中心值
centroids = estimator.cluster_centers_
# 聚类中心均值向量的总和
inertia = estimator.inertia_

print(lable_pred)
print(centroids)
print(inertia)