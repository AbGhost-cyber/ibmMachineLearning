from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load iris dataset
iris = load_iris()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Train KNN classifier and predict
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Calculate accuracy and standard deviation
acc = accuracy_score(y_test, y_pred)
std = np.std(y_pred == y_test) / np.sqrt(len(y_pred))

# Print results
print(acc)
print("Accuracy: {:.2f}%".format(acc * 100))
print("Standard deviation: {:.2f}".format(std))

if __name__ == '__main__':
    print()