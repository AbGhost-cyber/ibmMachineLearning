import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cust_df = pd.read_csv('../datasets/Cust_Segmentation.csv')

# PREPROCESSING
df = cust_df.drop(columns='Address')
# : indicates that we want to select all the rows
# while `1:` indicates that we only want to select the columns
# starting from the second column (index 1) up to the end of the array.
X = df.values[:, 1:]
X = np.nan_to_num(X)

clus_data_set = StandardScaler().fit_transform(X)

clusterNum = 3
k_means = KMeans(n_clusters=clusterNum, n_init=12)
k_means.fit(X)
labels = k_means.labels_

# we assign the labels to each row in the dataframe
df['Clus_km'] = labels
# print(df.head(n=5))
# df.groupby('Clus_km').mean()

# area = np.pi * (X[:, 1]) ** 2
# plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), alpha=0.7)
# plt.xlabel('Age', fontsize=18)
# plt.ylabel('Income', fontsize=16)

fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(float))
plt.show()

if __name__ == '__main__':
    print(X.dtype)
