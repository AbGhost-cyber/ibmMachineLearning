# Import the libraries we need to use in this lab
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import time
import warnings

warnings.filterwarnings('ignore')

raw_data = pd.read_csv('../datasets/creditcard.csv')

n_replicas = 10

big_raw_data = pd.DataFrame(np.repeat(raw_data.values, n_replicas, axis=0), columns=raw_data.columns)
# print("There are " + str(len(big_raw_data)) + " observations in the inflated credit card fraud dataset.")
# print("There are " + str(len(big_raw_data.columns)) + " variables in the dataset.")

# get the set of distinct classes
labels = big_raw_data.Class.unique()

# get the count of each class
sizes = big_raw_data.Class.value_counts().values

# data preprocessing such as scaling/normalization is typically useful for
# linear models to accelerate the training convergence

# standardize features by removing the mean and scaling to unit variance
big_raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(big_raw_data.iloc[:, 1: 30])
data_matrix = big_raw_data.values

# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = data_matrix[:, 1:30]

# y: labels vector
y = data_matrix[:, 30]

# data normalization
X = normalize(X, norm="l1")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# compute the sample weights to be used as input to the train routine so that
# it takes into account the class imbalance present in this dataset
w_train = compute_sample_weight('balanced', y_train)

# for reproducible output across multiple function calls, set random_state to a given integer value
decision_tree = DecisionTreeClassifier(max_depth=4, random_state=35)

# train a Decision Tree Classifier using scikit-learn
# run inference and compute the probabilities of the test samples
# to belong to the class of fraudulent transactions
# t0 = time.time()
decision_tree.fit(X_train, y_train, sample_weight=w_train)
y_pred = decision_tree.predict_proba(X_test)[:, 1]

# evaluate the Compute Area Under the Receiver Operating Characteristic
# Curve (ROC-AUC) score from the predictions
# sklearn_time = time.time() - t0
roc_auc = roc_auc_score(y_test, y_pred)

# print("Training time (s):  {0:.5f}".format(sklearn_time))

# plot the class value counts
# fig, ax = plt.subplots()
# ax.pie(sizes, labels=labels, autopct='%1.3f%%')
# ax.set_title('Target variable value counts')
# plt.show()

# PRACTICE: The credit card transactions have different amounts.
# Could you plot a histogram that shows the distribution of these amounts?
# What is the range of these amounts (min/max)? Could you print the 90th percentile of the amount values?
# plt.hist(big_raw_data.Amount.values, 6, histtype='bar', facecolor='g')
# plt.show()

if __name__ == '__main__':
    print()
    # print("Minimum amount value is ", np.min(big_raw_data.Amount.values))
    # print("Maximum amount value is ", np.max(big_raw_data.Amount.values))
    # print("90% of the transactions have an amount less or equal than ", np.percentile(raw_data.Amount.values, 90))
