import numpy as np
import matplotlib.pyplot as plt

class Autoencoder:
    def __init__(self, file_path, is_linear, hidden_layers, activation, epochs, lr):
        self.data = self.load_data(file_path)
        self.labels = self.data[:, 0]
        self.X = self.data[:, 1:]
        self.is_linear = is_linear
        self.hidden_layers = hidden_layers
        self.func, self.d_func = self.get_func(activation)
        self.epochs = epochs
        self.lr = lr
        self.W = []
        self.b = []
        self.losses = []

    def load_data(self, file_path):
        return np.genfromtxt(file_path, delimiter=',')

    def normalization(self):
        self.mean = np.mean(self.X, axis=0)
        self.std = np.std(self.X, axis=0)
        self.X_normalized = (self.X - self.mean) / self.std
        # self.range = np.max(self.X, axis=0) - np.min(self.X, axis=0)
        # self.X_normalized = (self.X - np.min(self.X, axis=0)) / (self.range)

    def get_func(self, name):
        if name == 'relu':
            alpha = 0
            return lambda x: np.where(x > 0, x, alpha * x), lambda x: np.where(x > 0, 1, alpha)
        elif name == 'tanh':
            return lambda x: np.tanh(x), lambda x: 1 - np.tanh(x) ** 2
        elif name == 'sigmoid':
            sigmoid = lambda x: 1 / (1 + np.exp(-x))
            return sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x))
        else:
            raise ValueError("Unsupported activation function")

    def initialize_W(self):
        layer_sizes = [self.X.shape[1]] + self.hidden_layers + [self.X.shape[1]]
        self.W = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1 for i in range(len(layer_sizes) - 1)]
        self.b = [np.zeros((1, layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]

    def forward(self, X):
        activations = [X]
        for i in range(len(self.W)):
            Z = np.dot(activations[-1], self.W[i]) + self.b[i]
            if i < len(self.W) - 1 and not self.is_linear:
                A = self.func(Z)
            else:
                A = Z
            activations.append(A)
        return activations

    def backward(self, activations, X):
        deltas = [activations[-1] - X]
        grads_w = []
        grads_b = []

        for i in reversed(range(len(self.W))):
            grad_w = np.dot(activations[i].T, deltas[-1]) / X.shape[0]
            grad_b = np.mean(deltas[-1], axis=0, keepdims=True)
            grads_w.append(grad_w)
            grads_b.append(grad_b)
            if i > 0:
                delta = np.dot(deltas[-1], self.W[i].T)
                if not self.is_linear:
                    delta *= self.d_func(activations[i])
                deltas.append(delta)

        grads_w.reverse()
        grads_b.reverse()
        return grads_w, grads_b

    def update_W(self, grads_w, grads_b):
        for i in range(len(self.W)):
            self.W[i] -= self.lr * grads_w[i]
            self.b[i] -= self.lr * grads_b[i]

    def train(self):
        self.initialize_W()
        for epoch in range(self.epochs):
            activations = self.forward(self.X_normalized)
            grads_w, grads_b = self.backward(activations, self.X_normalized)
            self.update_W(grads_w, grads_b)
            loss = np.mean((activations[-1] - self.X_normalized) ** 2)
            self.losses.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

    def visualize(self):
        reduced_data = self.forward(self.X_normalized)[len(self.hidden_layers)]
        for label in np.unique(self.labels):
            label_indices = np.where(self.labels == label)
            plt.scatter(
                reduced_data[label_indices, 0],
                reduced_data[label_indices, 1],
                label=f"Class {int(label)}"
            )
        if self.is_linear:
            plt.title("Linear Autoencoder")
        else:
            plt.title("Non-Linear Autoencoder")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.grid()
        plt.show()

    def reconstruct(self):
        activations = self.forward(self.X_normalized)
        self.reconstructed_data = activations[-1]
        reconstruction_error = np.mean((self.X_normalized - self.reconstructed_data) ** 2)
        print(f"Reconstruction Error: {reconstruction_error:.6f}")

    def plot_loss(self):
        plt.plot(self.losses, label="Training Loss")
        plt.title("Loss Function")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

    def run(self):
        self.normalization()
        self.train()
        self.visualize()
        self.reconstruct()
        self.plot_loss()


autoencoder = Autoencoder('wine.data', is_linear=False, hidden_layers=[2], activation='relu', epochs=5000, lr=0.01)
autoencoder.run()
