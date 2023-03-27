import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, f1_score

cell_df = pd.read_csv("../datasets/cell_samples.csv")

# ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue',
#                                                label='malignant')
# cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow',
#                                           label='benign', ax=ax)

# plt.show()

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype(int)

features_df = cell_df.drop(columns=['ID', 'Class'])
X = np.asarray(features_df)

# malignant or benign, currently 2, 4, so we need to change it to be binary
cell_df['Class'] = cell_df['Class'].astype(int)
y = np.asarray(cell_df['Class'])

# train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

yhat = clf.predict(X_test)

if __name__ == '__main__':
    print(confusion_matrix(y_test, yhat))
    print(classification_report(y_test, yhat))
    print(f1_score(y_test, yhat, average='weighted'))
