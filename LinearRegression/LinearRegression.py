import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, epochs, lr, is_normalization=False, is_standardize=False):
        self.epochs = epochs
        self.lr = lr
        self.is_normalization = is_normalization
        self.is_standardize = is_standardize
        self.W = np.random.rand(2, 1) * 0.05
        self.loss = []

    def normalization(self, X):
        _range = np.max(X) - np.min(X)
        return (X - np.min(X)) / _range

    def standardization(self, X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        return (X - mu) / sigma

    def judge_x(self, X):
        if self.is_normalization:
            X = self.normalization(X)
        if self.is_standardize:
            X = self.standardization(X)
        return X

    def _preprocess_x(self, X):
        X = self.judge_x(X)
        X_ = np.empty((100, 2))
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_

    def sgd_update(self, X, Y):
        data = np.empty((100, 3))
        data[:, 0:2] = X
        data[:, 2:] = Y
        for i in range(self.epochs):
            np.random.shuffle(data)
            X = data[:, 0:2]
            Y = data[:, 2:]
            Y_pred = X @ self.W
            loss = np.sum((Y - Y_pred) ** 2) / Y.size
            self.loss.append(loss)
            j = i % 99
            y_pred = np.reshape(Y_pred[j, :], (1, 1))
            y = np.reshape(Y[j, :], (1, 1))
            x = np.reshape(X[j, :], (1, 2))
            grad = (-(y - y_pred) @ x).T
            self.W = self.W - self.lr * grad

    def bgd_update(self, X, Y):
        for i in range(self.epochs):
            Y_pred = X @ self.W
            loss = np.sum((Y - Y_pred) ** 2) / Y.size
            self.loss.append(loss)
            grad = -np.dot(X.T, Y - Y_pred) / Y.size
            self.W = self.W - self.lr * grad

    def mbgd_update(self, X, Y):
        data = np.empty((100, 3))
        data[:, 0:2] = X
        data[:, 2:] = Y
        for i in range(self.epochs):
            np.random.shuffle(data)
            X = data[:, 0:2]
            Y = data[:, 2:]
            Y_pred = X @ self.W
            loss = np.sum((Y - Y_pred) ** 2) / Y.size
            self.loss.append(loss)
            for j in range(10):
                x = np.reshape(X[10 * j:10 * (j + 1)], (10, 2))
                y_pred = np.reshape(Y_pred[10 * j:10 * (j + 1)], (10, 1))
                y = np.reshape(Y[10 * j:10 * (j + 1)], (10, 1))
                grad = -np.dot(x.T, y - y_pred) / y.size
                self.W = self.W - self.lr * grad

    # 选择训练的方法
    def train(self, X, Y):
        X = self._preprocess_x(X)
        self.bgd_update(X, Y)
        # self.sgd_update(X, Y)
        # self.mbgd_update(X, Y)

    def plot_loss(self, X_train, Y_train):
        X_train = self.judge_x(X_train)
        plt.plot(self.loss)
        plt.grid(True)
        plt.show()

        plt.plot(X_train, Y_train)
        y_pred = self._preprocess_x(X_train) @ model.W
        plt.plot(X_train, y_pred)
        plt.show()


if __name__ == '__main__':
    X_train = np.arange(100).reshape(100, 1)
    a, b = 1, 10
    Y_train = a * X_train + b + np.random.normal(0, 5, size=X_train.shape)

    model = LinearRegression(200, 1e-5, False, False)
    model.train(X_train, Y_train)
    model.plot_loss(X_train, Y_train)
    print(f"Learned model: {model.W}")
