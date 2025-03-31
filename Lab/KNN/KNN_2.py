import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

data = np.loadtxt('wdbc.data', delimiter=',', dtype=str)


def standardize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma

X = data[:, 2:].astype(float)
X = standardize(X)
y = np.where(data[:, 1] == 'M', 1, 0)

repetitions = 20
k_values = range(1, 21)

results = {k: [] for k in k_values}

for _ in range(repetitions):

    idx = np.random.permutation(len(X))
    train_size = int(0.7 * len(X))
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    for k in k_values:
        correct = 0
        tree = KDTree(X_train)

        for i, x in enumerate(X_test):
            distances, indices = tree.query(x, k)
            if k == 1:
                indices = np.array([indices])
            labels = y_train[indices]
            if labels.ndim == 0:
                labels = np.array([labels])
            y_pred = np.bincount(labels).argmax()
            if y_pred == y_test[i]:
                correct += 1

        accuracy = correct / len(X_test)
        results[k].append(accuracy)

avg_accuracies=[]

for k in k_values:
    avg_accuracy = np.mean(results[k])
    avg_accuracies.append(avg_accuracy)
    print(f"k={k}: {avg_accuracy:.4f}")

plt.plot(k_values, avg_accuracies, marker='o')
plt.xlabel('k')
plt.ylabel('Test accuracy')
plt.show()
