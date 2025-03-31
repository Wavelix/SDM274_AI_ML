import numpy as np
import matplotlib.pyplot as plt


class MC:
    def __init__(self, layers, activation, lr, epochs, batch_size):
        self.layers = layers
        self.func = self.get_func(activation)
        self.d_func = self.get_d_func(activation)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.W = []
        self.b = []
        self.history = []

        for i in range(len(layers) - 1):
            self.W.append(np.random.randn(layers[i], layers[i + 1]) * 0.01)
            self.b.append(np.zeros((1, layers[i + 1])))

    def get_func(self, name):
        if name == 'relu':
            return lambda x: np.maximum(0, x)
        elif name == 'tanh':
            return lambda x: np.tanh(x)
        elif name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function")

    def get_d_func(self, name):
        if name == 'relu':
            return lambda x: (x > 0).astype(float)
        elif name == 'tanh':
            return lambda x: 1 - np.tanh(x) ** 2
        elif name == 'sigmoid':
            return lambda x: self.get_func('sigmoid')(x) * (
                        1 - self.get_func('sigmoid')(x))
        else:
            raise ValueError("Unsupported activation function")

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, x):
        a = x
        activations = [a]
        zs = []
        for w, b in zip(self.W, self.b):
            z = np.dot(a, w) + b
            zs.append(z)
            a = self.softmax(z) if w is self.W[-1] else self.func(z)
            activations.append(a)
        return activations, zs

    def backward(self, activations, zs, y):
        m = y.shape[0]
        y_onehot = np.eye(self.layers[-1])[y]
        dz = activations[-1] - y_onehot
        dw = np.dot(activations[-2].T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m

        grads_w = [dw]
        grads_b = [db]

        for l in range(len(self.layers) - 2, 0, -1):
            dz = np.dot(dz, self.W[l].T) * self.d_func(zs[l - 1])
            dw = np.dot(activations[l - 1].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            grads_w.insert(0, dw)
            grads_b.insert(0, db)

        return grads_w, grads_b

    def update(self, grad_W, grad_b):
        for i in range(len(self.W)):
            self.W[i] -= self.lr * grad_W[i]
            self.b[i] -= self.lr * grad_b[i]

    def get_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        y_onehot = np.eye(self.layers[-1])[y_true]
        loss = -np.sum(y_onehot * np.log(y_pred + 1e-9)) / m
        return loss

    def train(self, X, y):
        m = X.shape[0]
        for epoch in range(self.epochs):

            indices = np.arange(m)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            epoch_loss = 0

            for i in range(0, m, self.batch_size):

                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                activations, zs = self.forward(X_batch)
                loss = self.get_loss(activations[-1], y_batch)
                epoch_loss += loss

                grad_W, grad_b = self.backward(activations, zs, y_batch)

                self.update(grad_W, grad_b)

            epoch_loss /= (m // self.batch_size)
            self.history.append(epoch_loss)

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")

        plt.plot(self.history)
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def accuracy(self, y_pred, y_true):
        return np.mean(y_pred == y_true)

train_data = np.loadtxt('optdigits.tra', delimiter=',')
test_data = np.loadtxt('optdigits.tes', delimiter=',')

X_train, y_train = train_data[:, :-1], train_data[:, -1].astype(int)
X_test, y_test = test_data[:, :-1], test_data[:, -1].astype(int)

model = MC(layers=[64,64,10], activation='relu', lr=0.01, epochs=500, batch_size=64)
model.train(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = model.accuracy(y_pred, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")
