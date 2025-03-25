import numpy as np


class AdaBoost:
    def __init__(self):
        self.alphas = []
        self.stumps = []

    def load_data(self, filepath):
        data = np.loadtxt(filepath)
        X = data[:, :-1]
        y = data[:, -1]
        mask = (y != 2)
        X, y = X[mask], y[mask]
        y = np.where(y == 1, 1, -1)
        return X, y

    def split_data(self, X, y, ratio=0.7):
        unique_labels = np.unique(y)
        train_indices = []
        test_indices = []

        for label in unique_labels:
            label_indices = np.where(y == label)[0]
            np.random.shuffle(label_indices)
            split = int(ratio * len(label_indices))
            train_indices.extend(label_indices[:split])
            test_indices.extend(label_indices[split:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        return X[train_indices], y[train_indices], X[test_indices], y[test_indices]

    def _stump(self, X, y, w):
        m, n = X.shape
        best_stump = {}
        best_error = float('inf')
        best_predictions = None

        for feature in range(n):
            feature_values = np.unique(X[:, feature])
            for threshold in feature_values:
                for inequality in ['lt', 'gt']:
                    predictions = np.ones(m)
                    if inequality == 'lt':
                        predictions[X[:, feature] <= threshold] = -1
                    else:
                        predictions[X[:, feature] > threshold] = -1

                    error = np.sum(w[predictions != y])
                    if error < best_error:
                        best_error = error
                        best_predictions = predictions.copy()
                        best_stump = {
                            'feature': feature,
                            'threshold': threshold,
                            'inequality': inequality
                        }

        return best_stump, best_error, best_predictions

    def fit(self, X, y, num_iter=50):
        m, n = X.shape
        w = np.ones(m) / m
        for _ in range(num_iter):
            stump, error, predictions = self._stump(X, y, w)

            alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))

            w = w * np.exp(-alpha * y * predictions)
            w /= np.sum(w)

            self.stumps.append(stump)
            self.alphas.append(alpha)

    def predict(self, X):
        m = X.shape[0]
        final_predictions = np.zeros(m)
        for alpha, stump in zip(self.alphas, self.stumps):
            predictions = np.ones(m)
            feature = stump['feature']
            threshold = stump['threshold']
            inequality = stump['inequality']

            if inequality == 'lt':
                predictions[X[:, feature] <= threshold] = -1
            else:
                predictions[X[:, feature] > threshold] = -1

            final_predictions += alpha * predictions
        return np.sign(final_predictions)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)


if __name__ == "__main__":
    total_accuracy = []
    model = AdaBoost()
    for _ in range(50):
        X, y = model.load_data("seeds_dataset.txt")
        X_train, y_train, X_test, y_test = model.split_data(X, y, ratio=0.7)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        total_accuracy.append(model.accuracy(y_test, y_pred))
    print("Accuracy:", np.mean(total_accuracy), "Var: ", np.var(total_accuracy))
