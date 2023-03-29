import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, jaccard_score, f1_score, \
    log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

df = pd.read_csv("datasets/Weather_Data.csv")

# PREPROCESSING
df_sydney_processed = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df_sydney_processed.replace(['No', 'Yes'], [0, 1], inplace=True)

# TRAINING & TEST
df_sydney_processed.drop(columns='Date', inplace=True)
features = df_sydney_processed.drop(columns='RainTomorrow', axis=1)
Y = df_sydney_processed['RainTomorrow']

x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=10)
LinearReg = LinearRegression()
LinearReg.fit(x_train, y_train)
predictions = LinearReg.predict(x_test)

# EVALUATION
LinearRegression_MAE = mean_absolute_error(y_test, predictions)
LinearRegression_MSE = mean_squared_error(y_test, predictions)
LinearRegression_R2 = r2_score(y_test, predictions)

Report = pd.DataFrame(
    {'Metric': ['MAE', 'MSE', 'R2'],
     'Value': [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2]})

# USING KNN
KNN = KNeighborsClassifier(n_neighbors=4)
KNN.fit(x_train, y_train)
predictions = KNN.predict(x_test)

# EVALUATION
KNN_Accuracy_Score = accuracy_score(y_test, predictions)
KNN_JaccardIndex = jaccard_score(y_test, predictions)
KNN_F1_Score = f1_score(y_test, predictions)

# USING DECISION TREE
Tree = DecisionTreeClassifier()
Tree.fit(x_train, y_train)
predictions = Tree.predict(x_test)

# EVALUATION
Tree_Accuracy_Score = accuracy_score(y_test, predictions)
Tree_JaccardIndex = jaccard_score(y_test, predictions)
Tree_F1_Score = f1_score(y_test, predictions)

# LOGISTIC REGR
x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=1)
LR = LogisticRegression(solver='liblinear')
LR.fit(x_train, y_train)
predictions = LR.predict(x_test)
LR_Accuracy_Score = accuracy_score(y_test, predictions)
LR_JaccardIndex = jaccard_score(y_test, predictions)
LR_F1_Score = f1_score(y_test, predictions)
LR_Log_Loss = log_loss(y_test, predictions)

# SVM
SVM = SVC()
SVM.fit(x_train, y_train)
predictions = SVM.predict(x_test)
SVM_Accuracy_Score = accuracy_score(y_test, predictions)
SVM_JaccardIndex = jaccard_score(y_test, predictions)
SVM_F1_Score = f1_score(y_test, predictions)

m_report = pd.DataFrame(
    {'Algorithm': ['KNN', 'Decision Tree', 'Logistic Regression', 'SVM'],
     'Accuracy': [KNN_Accuracy_Score, Tree_Accuracy_Score, LR_Accuracy_Score, SVM_Accuracy_Score],
     'Jaccard Index': [KNN_JaccardIndex, Tree_JaccardIndex, LR_JaccardIndex, SVM_JaccardIndex],
     'F1-Score': [KNN_F1_Score, Tree_F1_Score, LR_F1_Score, SVM_F1_Score],
     'LogLoss': ["--", '--', LR_Log_Loss, '--']})

if __name__ == '__main__':
    print(m_report)
