import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, jaccard_score, classification_report, log_loss
import matplotlib.pyplot as plt
import itertools

churn_df = pd.read_csv('../datasets/ChurnData.csv')
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]

X = np.asanyarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asanyarray(churn_df['churn'])

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# C specifies the regularization parameter that controls the strength of the regularization applied to the model
#  A smaller value of `C` indicates a stronger regularization effect,
#  which can help to prevent overfitting the training data.
# `solver='liblinear'` specifies the algorithm used to solve the optimization problem
# that arises in logistic regression. In this case 'liblinear' is used which is a good choice for small datasets.
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
y_hat = LR.predict(X_test)
y_hat_prob = LR.predict_proba(X_test)

# we can define jaccard as the size of the intersection divided by the size of the union of the two label sets.
# If the entire set of predicted labels for a sample strictly matches with the true set of labels,
# then the subset accuracy is 1.0; otherwise it is 0.0.
my_jaccard_score = jaccard_score(y_test, y_hat, pos_label=0)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':
    print("jaccard score is: %.2f" % my_jaccard_score)
    print(confusion_matrix(y_test, y_hat))
    print(classification_report(y_test, y_hat))
    print(log_loss(y_test, y_hat_prob))
    # Compute confusion matrix
    # cnf_matrix = confusion_matrix(y_test, y_hat, labels=[1, 0])
    # np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=['churn=1', 'churn=0'], normalize=False, title='Confusion matrix')
