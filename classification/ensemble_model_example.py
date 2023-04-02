import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

bc_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                    'breast-cancer-wisconsin/wdbc.data', header=None)

# Split the dataset into X (features) and y (target)
X = bc_df.iloc[:, 2:].values
y = bc_df.iloc[:, 1].values
# Preprocess the data
X = StandardScaler().fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

# next we define our individual models
lr_model = LogisticRegression(random_state=3)
svm_model = SVC(random_state=3)
knn_model = KNeighborsClassifier(n_neighbors=4)

# we will then create a list of models to pass to our ensemble model
models = [('lr', lr_model), ('svm', svm_model), ('knn', knn_model)]
ensemble_model = VotingClassifier(estimators=models, voting='hard')

# Train and evaluate the individual models
kfold = KFold(n_splits=5, shuffle=True, random_state=3)
for name, model in models:
    scores = cross_val_score(model, X_train, y_train, cv=kfold)
    print(f'{name}: Cross-validation accuracy: {scores.mean():.2f} +/- {scores.std():.2f}')

# Train and evaluate the ensemble model
ensemble_model.fit(X_train, y_train)
# y_hat = ensemble_model.predict(X_test)
ensemble_acc = ensemble_model.score(X_test, y_test)
print(f'Ensemble model accuracy: {ensemble_acc:.2f}')
# print(confusion_matrix(y_test, y_hat))

if __name__ == '__main__':
    print()
