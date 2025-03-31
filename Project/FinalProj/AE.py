import numpy as np
import matplotlib.pyplot as plt


class Autoencoder:
    def __init__(self, input_dim, hidden_layer, lr=0.01, epochs=1500, type=2):
        self.input_dim = input_dim
        self.hidden_layer = hidden_layer
        self.lr = lr
        self.epochs = epochs
        self.type = type

        self.W = []
        self.b = []
        layer_dims = [input_dim] + hidden_layer + [input_dim]
        for i in range(len(layer_dims) - 1):
            w = np.random.randn(layer_dims[i], layer_dims[i + 1]) * np.sqrt(2. / (layer_dims[i] + layer_dims[i + 1]))
            b = np.zeros((1, layer_dims[i + 1]))
            self.W.append(w)
            self.b.append(b)
        self.loss_history = []

    def func(self, z):
        # return np.tanh(z)
        return np.where(z > 0, z, 0.01 * z)

    def d_func(self, z):
        # return 1 - np.tanh(z) ** 2
        return np.where(z > 0, 1, 0.01)

    def forward(self, X):
        activations = [X]
        zs = []
        for w, b in zip(self.W[:-1], self.b[:-1]):
            z = np.dot(activations[-1], w) + b
            zs.append(z)
            activations.append(self.func(z))
        z = np.dot(activations[-1], self.W[-1]) + self.b[-1]
        zs.append(z)
        activations.append(z)
        return activations, zs

    def backward(self, X, activations, zs):
        m = X.shape[0]
        grads_w = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        delta = activations[-1] - X
        grads_w[-1] = np.dot(activations[-2].T, delta) / m
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / m
        for l in range(len(self.W) - 2, -1, -1):
            delta = np.dot(delta, self.W[l + 1].T) * self.d_func(zs[l])
            grads_w[l] = np.dot(activations[l].T, delta) / m
            grads_b[l] = np.sum(delta, axis=0, keepdims=True) / m
        return grads_w, grads_b

    def train(self, X):
        for epoch in range(self.epochs):
            activations, zs = self.forward(X)
            grads_w, grads_b = self.backward(X, activations, zs)
            for i in range(len(self.W)):
                self.W[i] -= self.lr * grads_w[i]
                self.b[i] -= self.lr * grads_b[i]
            loss = np.mean((X - activations[-1]) ** 2)
            self.loss_history.append(loss)

            # if epoch % 10000 == 0:
            #     print(loss)

    def encode(self, X):
        for w, b in zip(self.W[:-1], self.b[:-1]):
            X = self.func(np.dot(X, w) + b)
        return X

    def decode(self, X_encoded):
        for w, b in zip(self.W[len(self.hidden_layer):], self.b[len(self.hidden_layer):]):
            X_encoded = np.dot(X_encoded, w) + b
        return X_encoded

    def reconstruction_error(self, X, X_reconstructed):
        return np.mean((X - X_reconstructed) ** 2)

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title("Training Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()


def load_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return (X - X.mean(axis=0)) / X.std(axis=0), y
    # return (X - X.mean(axis=0)), y


def plot_2d(X, y):
    plt.figure()
    for label in np.unique(y):
        plt.scatter(X[y == label, 0], X[y == label, 1], label=f"Class {label}")
    plt.legend()
    plt.title("2D Visualization")
    plt.show()


def plot_3d(X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label in np.unique(y):
        ax.scatter(X[y == label, 0], X[y == label, 1], X[y == label, 2], label=f"Class {label}")
    ax.legend()
    plt.title("3D Visualization")
    plt.show()


if __name__ == "__main__":
    file_path = "seeds_dataset.txt"
    X, y = load_data(file_path)

    # 降维到2D
    AE_2d = Autoencoder(input_dim=X.shape[1], hidden_layer=[2], lr=0.04, epochs=200000, type=2)
    AE_2d.train(X)
    X_encoded_2d = AE_2d.encode(X)
    X_reconstructed_2d = AE_2d.decode(X_encoded_2d)
    error_2d = AE_2d.reconstruction_error(X, X_reconstructed_2d)
    print(f"Reconstruction Error (2D): {error_2d:.6f}")
    plot_2d(X_encoded_2d, y)

    # 降维到3D
    AE_3d = Autoencoder(input_dim=X.shape[1], hidden_layer=[3], lr=0.02, epochs=200000, type=3)
    AE_3d.train(X)
    X_encoded_3d = AE_3d.encode(X)
    X_reconstructed_3d = AE_3d.decode(X_encoded_3d)
    error_3d = AE_3d.reconstruction_error(X, X_reconstructed_3d)
    print(f"Reconstruction Error (3D): {error_3d:.6f}")
    plot_3d(X_encoded_3d, y)
