import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

data = np.loadtxt('wdbc.data', delimiter=',', dtype=str)


def standardize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma

X = data[:, 2:].astype(float)
X = standardize(X)
y = np.where(data[:, 1] == 'M', 1, 0)

k_values = range(1, 11)

idx = np.random.permutation(len(X))
train_size = int(0.7 * len(X))
train_idx = idx[:train_size]
test_idx = idx[train_size:]

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]


def knn(X_train, y_train, X_test, y_test):
    scores = []

    for k in k_values:
        score = 0
        tree = KDTree(X_train)

        for i, x in enumerate(X_test):
            distances, indices = tree.query(x, k)
            if k == 1:
                indices = np.array(indices)
            labels = y_train[indices]
            if labels.ndim == 0:
                labels = np.array([labels])
            y_pred = np.bincount(labels).argmax()
            score += 1 if y_pred == y_test[i] else 0
        mean_score = score / len(y_test)
        scores.append(mean_score)

    return scores


scores = knn(X_train, y_train, X_test, y_test)

for k, score in zip(k_values, scores):
    print(f"k={k}, score={score:.4f}")

plt.plot(k_values, scores, marker='o')
plt.xlabel('k')
plt.ylabel('Test accuracy')
plt.show()
