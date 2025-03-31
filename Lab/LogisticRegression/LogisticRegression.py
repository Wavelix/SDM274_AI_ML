import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, epochs, lr, is_normalization, is_standardization):
        self.epochs = epochs
        self.lr = lr
        self.is_normalization = is_normalization
        self.is_standardization = is_standardization

        self.loss_history = []
        self.W = np.random.rand(14, 1) * 0.05
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0

    def standardize(self, X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        return (X - mu) / sigma

    def normalize(self, X):
        _range = np.max(X, axis=0) - np.min(X, axis=0)
        return X / _range

    def preprocess_X(self, X):
        if self.is_normalization:
            X = self.normalize(X)
        elif self.is_standardization:
            X = self.standardize(X)
        m, n = X.shape
        X_ = np.empty((m, n + 1))
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_

    def active(self, X_):
        z = X_ @ self.W
        out = 1. / (1. + np.exp(-z))
        return out

    def _loss(self, t, y):
        epsilon = 1e-7
        loss = -np.sum(t * np.log(y + epsilon)) / 91 - np.sum((1 - t) * np.log(1 - y + epsilon)) / 91
        return loss

    def shuffle_data(self, X_, t):
        data_shuffle = np.empty([91, 15])
        data_shuffle[:, :14] = X_
        data_shuffle[:, 14:] = t
        np.random.shuffle(data_shuffle)
        X_ = data_shuffle[:, :14]
        t = data_shuffle[:, 14:]
        return X_, t

    def sgd_update(self, X, t):
        X_ = self.preprocess_X(X)

        for i in range(self.epochs):
            X_, t = self.shuffle_data(X_, t)
            y = self.active(X_)
            self.loss_history.append(self._loss(t, y))
            j = i % 91
            x_ = np.reshape(X_[j, :], (1, 14))
            y_ = np.reshape(y[j, :], (1, 1))
            t_ = np.reshape(t[j, :], (1, 1))
            grad = -((t_ - y_) @ x_).T
            self.W -= self.lr * grad

    def mbgd_update(self, X, t):
        X_ = self.preprocess_X(X)
        X_, t = self.shuffle_data(X_, t)
        for i in range(self.epochs):
            y = self.active(X_)
            self.loss_history.append(self._loss(t, y))
            for j in range(9):
                x_ = np.reshape(X_[10 * j:10 * (j + 1), :], (10, 14))
                y_ = np.reshape(y[10 * j:10 * (j + 1), :], (10, 1))
                t_ = np.reshape(t[10 * j:10 * (j + 1), :], (10, 1))
                grad = -x_.T @ (t_ - y_)/t_.size
                self.W -= self.lr * grad

    def predict(self, X_test, t_test):
        X_ = self.preprocess_X(X_test)
        y_pred = np.where(self.active(X_) > 0.5, 1, 0)
        for i in range(X_test.shape[0]):
            if t_test[i, 0] == 1 and y_pred[i, 0] == 1:
                self.TP += 1
            elif t_test[i, 0] == 1 and y_pred[i, 0] == 0:
                self.FN += 1
            elif t_test[i, 0] == 0 and y_pred[i, 0] == 1:
                self.FP += 1
            elif t_test[i, 0] == 0 and y_pred[i, 0] == 0:
                self.TN += 1
        A = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        R = self.TP / (self.TP + self.FN)
        P = self.TP / (self.TP + self.FP)
        F1 = 2 * (P * R) / (P + R)
        print(f"Accuracy: {A:.2f} , Recall: {R:.2f} , Precision:{P:.2f} , F1: {F1:.2f}")
        x = np.linspace(0, 39, 39)
        plt.scatter(x, y_pred, color='red', marker='x', label='false prediction')
        plt.scatter(x, t_test, color='blue', label='target')
        plt.title("Performance on Test Set")
        plt.legend()
        plt.show()

    def plot_img(self, X_train, t_train):
        plt.plot(self.loss_history)
        plt.title("Loss Curve")
        plt.grid(True)
        plt.show()
        X_ = self.preprocess_X(X_train)
        y_pred = np.where(self.active(X_) > 0.5, 1, 0)
        x = np.linspace(0, 91, 91)
        plt.scatter(x, y_pred, color='red', marker='x', label='false prediction')
        plt.title("Performance on Training Set")
        plt.scatter(x, t_train, color='blue', label='target')
        plt.legend()
        plt.show()

    def read_data(self):
        try:
            file = 'wine.data'
            data = np.genfromtxt(file, delimiter=',')
            wine_data = np.empty([130, 14])
            wine_data[:, :] = data[:130, :]
            wine_data[59:, :1] = 0
            return wine_data
        except IOError:
            print(f"The file 'wine.data' cannot be opened")
            return None

    def preprocess_data(self, wine_data):
        class_1 = wine_data[:59, :]
        class_2 = wine_data[59:, :]
        np.random.shuffle(class_1)
        np.random.shuffle(class_2)
        train_data_1 = class_1[:41, :]
        train_data_2 = class_2[:50, :]
        test_data_1 = class_1[41:, :]
        test_data_2 = class_2[50:, :]
        train_data = np.empty([91, 14])
        train_data[:41, :] = train_data_1
        train_data[41:, :] = train_data_2
        test_data = np.empty([39, 14])
        test_data[:18, :] = test_data_1
        test_data[18:, :] = test_data_2
        return train_data, test_data


if __name__ == '__main__':
    model = LogisticRegression(2000, 1e-1, False, True)
    wine_data = model.read_data()
    train_data, test_data = model.preprocess_data(wine_data)

    X_train = train_data[:, 1:]
    t_train = train_data[:, :1]
    X_test = test_data[:, 1:]
    t_test = test_data[:, :1]

    gd_method = input("Choose method for gradient descent, sgd or mbgd: [s/m]\n")
    if gd_method == "s":
        model.sgd_update(X_train, t_train)
    elif gd_method == "m":
        model.mbgd_update(X_train, t_train)
    else:
        print("Invalid method, sgd will be applied")

    model.plot_img(X_train, t_train)
    model.predict(X_test, t_test)
