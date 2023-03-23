import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# CLASSIFICATION EXAMPLE USING K-NEAREST NEIGHBOURS

df = pd.read_csv('teleCust1000t.csv')

# print(df['income'].value_counts())
# df.hist(column='income', bins=50)
# plt.show()

# SELECT FEATURE SET
X = df.drop(columns=['custcat']).values
y = df['custcat'].values

# NORMALIZE DATA
# Data Standardization gives the data zero mean and unit variance,
# it is good practice, especially for algorithms such as KNN which is based on the distance of data points:
scaler = StandardScaler()
X = scaler.fit_transform(X.astype(float))

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
# print('Train set: ', X_train.shape, y_train.shape)
# print('Test set: ', X_test.shape, y_test.shape)

# TRAINING
k = 11
knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
y_pred = knn.predict(X_test)

# # experimenting best K
# Ks = 1
# # stores accuracy of the model's performance
# mean_acc = np.zeros((Ks - 1))
# # stores standard deviation of the model's performance
# std_acc = np.zeros((Ks - 1))
#
# for n in range(1, Ks):
#     # Train model and predict
#     knn = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#     mean_acc[n - 1] = accuracy_score(y_test, y_pred)
#
#     std_acc[n - 1] = np.std(y_pred == y_test) / np.sqrt(y_pred.shape[0])

# plot model accuracy for n neighbors
# plt.plot(range(1,Ks),mean_acc,'g')
# plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
# plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
# plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
# plt.ylabel('Accuracy ')
# plt.xlabel('Number of Neighbors (K)')
# plt.tight_layout()
# plt.show()

if __name__ == '__main__':
    # print("train set accuracy: ", accuracy_score(y_train, knn.predict(X_train)))
    print("Test set accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
