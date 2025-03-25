import numpy as np
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, layers, lr=0.01, epochs=100):
        self.layers = layers
        self.lr = lr
        self.epochs = epochs
        self.W = []
        self.b = []
        self.loss_history = []

        for i in range(len(layers) - 1):
            weight = np.random.randn(layers[i], layers[i + 1]) * 0.01
            bias = np.zeros((1, layers[i + 1]))
            self.W.append(weight)
            self.b.append(bias)

    def relu(self, z):
        return np.maximum(0, z)

    def d_relu(self, z):
        return np.where(z > 0, 1, 0)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        activations = [X]
        zs = []

        for w, b in zip(self.W[:-1], self.b[:-1]):
            z = np.dot(activations[-1], w) + b
            zs.append(z)
            activation = self.relu(z)
            activations.append(activation)

        # 输出层
        z = np.dot(activations[-1], self.W[-1]) + self.b[-1]
        zs.append(z)
        activations.append(self.softmax(z))

        return activations, zs

    def backward(self, X, y, activations, zs):
        m = X.shape[0]
        grads_w = [None] * len(self.W)
        grads_b = [None] * len(self.b)

        # 输出层误差
        delta = activations[-1] - y
        grads_w[-1] = np.dot(activations[-2].T, delta) / m
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / m

        # 反向传播隐藏层
        for l in range(len(self.layers) - 2, 0, -1):
            delta = np.dot(delta, self.W[l].T) * self.d_relu(zs[l - 1])
            grads_w[l - 1] = np.dot(activations[l - 1].T, delta) / m
            grads_b[l - 1] = np.sum(delta, axis=0, keepdims=True) / m

        return grads_w, grads_b

    def train(self, X, y):
        for epoch in range(self.epochs):
            activations, zs = self.forward(X)
            grads_w, grads_b = self.backward(X, y, activations, zs)

            for i in range(len(self.W)):
                self.W[i] -= self.lr * grads_w[i]
                self.b[i] -= self.lr * grads_b[i]

            loss = -np.sum(y * np.log(activations[-1] + 1e-8)) / X.shape[0]
            self.loss_history.append(loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()


def load_data(file_path, classes):
    data = np.loadtxt(file_path)
    if classes == 3:
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        # 将标签转换为独热编码
        num_classes = len(np.unique(y))
        y_one_hot = np.zeros((y.shape[0], num_classes))
        y_one_hot[np.arange(y.shape[0]), y - 1] = 1
        return X, y, y_one_hot
    elif classes == 2:
        filtered_data = data[data[:, -1] != 2]
        filtered_data = filtered_data[:140]
        X = filtered_data[:, :-1]
        y = filtered_data[:, -1].astype(int)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        # 将标签转换为独热编码
        num_classes = len(np.unique(y))
        y_one_hot = np.zeros((y.shape[0], num_classes))
        label = np.where(y == 1, 0, 1)
        y_one_hot[np.arange(y.shape[0]), label] = 1
        return X, y, y_one_hot


def split_data(X, y, y_one_hot, test_size=0.3):
    X_train, X_test = [], []
    y_train, y_test = [], []

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        np.random.shuffle(idx)

        split = int(len(idx) * (1 - test_size))
        X_train.append(X[idx[:split]])
        X_test.append(X[idx[split:]])
        y_train.append(y_one_hot[idx[:split]])
        y_test.append(y_one_hot[idx[split:]])

    return np.vstack(X_train), np.vstack(X_test), np.vstack(y_train), np.vstack(y_test)


if __name__ == "__main__":
    file_path = "seeds_dataset.txt"
    classes = 2

    X, y, y_one_hot = load_data(file_path, classes)
    X_train, X_test, y_train, y_test = split_data(X, y, y_one_hot)
    model = MLP(layers=[7, 10, classes], lr=0.05, epochs=50000)
    model.train(X_train, y_train)

    y_pred = model.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)

    acc = model.accuracy(y_test_labels, y_pred)
    print(f"Accuracy: {acc:.4f}")

    model.plot_loss()
