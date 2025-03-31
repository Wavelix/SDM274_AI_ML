import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, epochs, lr, is_normalization, is_standardization):
        self.epochs = epochs
        self.lr = lr
        self.W = np.random.rand(14, 1) * 0.05
        self.loss = []
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
        self.is_normalization = is_normalization
        self.is_standardization = is_standardization

    def standarlization(self, X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        return (X - mu) / sigma
    def normalization(self, X):
        _range=np.max(X,axis=0) - np.min(X,axis=0)
        return (X-np.min(X,axis=0)) / _range

    def _preprocess_X(self, X):
        if self.is_normalization:
            X = self.normalization(X)
        elif self.is_standardization:
            X = self.standarlization(X)
        m, n = X.shape
        X_ = np.empty([m, n + 1])
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_

    def sgd_update(self, X_, t):
        data = np.empty([91, 15])
        data[:, :14] = X_
        data[:, 14:] = t
        np.random.shuffle(data)
        for i in range(self.epochs):
            X_ = data[:, :14]
            t = data[:, 14:]
            y = X_ @ self.W
            j = i % 91
            x_ = np.reshape(X_[j, :], (1, 14))
            y_ = np.reshape(y[j, :], (1, 1))
            t_ = np.reshape(t[j, :], (1, 1))
            _loss = 0
            for k in range(91):
                _loss += -t[k, 0] * y[k, 0] if t[k, 0] * y[k, 0] < 0 else 0
            _loss = _loss / 91
            self.loss.append(_loss)
            grad = -(t_ @ x_).T if t_.T @ y_ < 0 else 0
            self.W = self.W - self.lr * grad

    def bgd_update(self, X_, t):
        for i in range(self.epochs):
            y = X_ @ self.W
            _loss, grad = 0, 0
            _zero = np.zeros(14)
            for j in range(91):
                _loss += -t[j, 0] * y[j, 0] if t[j, 0] * y[j, 0] < 0 else 0
                grad += -X_[j, :].T * t[j, 0] if t[j, 0] * y[j, 0] < 0 else _zero
            _loss = _loss / 91

            self.loss.append(_loss)
            grad = np.reshape(grad, (14, 1)) / 91
            self.W = self.W - self.lr * grad

    def sgd_train(self, X_train, t_train):
        X_ = self._preprocess_X(X_train)
        self.sgd_update(X_, t_train)

    def bgd_train(self, X_train, t_train):
        X_ = self._preprocess_X(X_train)
        self.bgd_update(X_, t_train)

    def predict(self, X_test, t_test):
        X_ = self._preprocess_X(X_test)
        y_pre = X_ @ self.W
        for i in range(39):
            if y_pre[i, 0] >= 0:
                y_pre[i, 0] = 1
            else:
                y_pre[i, 0] = -1
        for i in range(X_test.shape[0]):
            if t_test[i, 0] == 1 and y_pre[i, 0] > 0:
                self.TP += 1
            elif t_test[i, 0] == 1 and y_pre[i, 0] < 0:
                self.FN += 1
            elif t_test[i, 0] == -1 and y_pre[i, 0] > 0:
                self.FP += 1
            elif t_test[i, 0] == -1 and y_pre[i, 0] < 0:
                self.TN += 1

        A = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        R = self.TP / (self.TP + self.FN)
        P = self.TP / (self.TP + self.FP)
        F1 = 2 * (P * R) / (P + R)
        print(f"Accuracy: {A:.2f} , Recall: {R:.2f} , Precision:{P:.2f} , F1: {F1:.2f}")

        x = np.linspace(0, 39, 39)
        plt.scatter(x, y_pre, color='red')
        plt.plot(t_test)
        plt.title("Performance on Test Set")
        plt.grid(True)
        plt.show()

    def plot_img(self, X_train, t_train):
        plt.plot(self.loss)
        plt.title("Loss Curve")
        plt.grid(True)
        plt.show()

        X_ = self._preprocess_X(X_train)
        y_pre = X_ @ self.W
        for i in range(91):
            if y_pre[i, 0] >= 0:
                y_pre[i, 0] = 1
            else:
                y_pre[i, 0] = -1

        x = np.linspace(0, 91, 91)
        plt.scatter(x, y_pre, color='red')
        plt.title("Performance on Training Set")
        plt.plot(t_train)
        plt.show()

    def read_data(self):
        try:
            file = 'wine.data'
            data = np.genfromtxt(file, delimiter=',')
            wine_data = np.empty([130, 14])
            wine_data[:, :] = data[:130, :]
            wine_data[59:, :1] = -1
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
    model = Perceptron(1000, 1e-3,False,True)
    wine_data = model.read_data()
    train_data, test_data = model.preprocess_data(wine_data)

    X_train = train_data[:, 1:]
    t_train = train_data[:, :1]
    X_test = test_data[:, 1:]
    t_test = test_data[:, :1]

    gd_method = input("Choose method for gradient descent: sgd/bgd\n")
    if gd_method == "sgd":
        model.sgd_train(X_train, t_train)
    elif gd_method == "bgd":
        model.bgd_train(X_train, t_train)
    else:
        print("Invalid method, sgd will be applied")

    model.plot_img(X_train, t_train)
    model.predict(X_test, t_test)
