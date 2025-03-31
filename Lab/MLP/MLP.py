import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class MLP:
    def __init__(self, layers, task='regression'):
        self.layer_sizes = layers
        self.task = task
        self.W = []
        self.b = []
        self.batch_error=[]

        for i in range(len(layers) - 1):
            self.W.append(np.random.randn(layers[i], layers[i + 1]) )
            self.b.append(np.zeros((1, layers[i + 1])))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, x):
        return x * (1 - x)

    def ReLU(self, x):
        return np.maximum(0, x)

    def d_ReLU(self, x):
        return np.where(x > 0, 1, 0)

    def Tanh(self, x):
        return np.tanh(x)

    def d_Tanh(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, X):
        self.activations = [X]
        for i in range(len(self.W) - 1):
            X = self.sigmoid(np.dot(X, self.W[i]) + self.b[i])
            self.activations.append(X)

        X = self.sigmoid(np.dot(X, self.W[-1]) + self.b[-1])
        self.activations.append(X)
        return X

    def backward(self, y, lr=0.01):
        m = y.shape[0]
        y = y.reshape(-1, 1)

        error = (self.activations[-1] - y) / (self.activations[-1] * (1 - self.activations[-1]))
        deltas = [error * self.d_sigmoid(self.activations[-1])]

        for i in range(len(self.activations) - 2, 0, -1):
            delta = deltas[-1].dot(self.W[i].T) * self.d_sigmoid(self.activations[i])
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.W)):
            self.W[i] -= lr * self.activations[i].T.dot(deltas[i]) / m
            self.b[i] -= lr * np.sum(deltas[i], axis=0, keepdims=True) / m

    def predict(self, X):
        y_hat = self.forward(X)
        if self.task == 'classification':
            return (y_hat > 0.5).astype(int)
        return y_hat

    def compute_loss(self, y, y_hat):
        m = y.shape[0]
        return -np.sum(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8)) / m


def train(model_class, layers, X, y, task='regression', k=5, epochs=1000, lr=0.01, batch_size=20):
    n_samples = X.shape[0]
    index = np.arange(n_samples)
    fold_size = n_samples // k
    train_losses = []
    val_losses = []

    model = model_class(layers, task=task)

    for epoch in range(epochs):
        epoch_train_loss = []
        epoch_val_loss = []
        np.random.shuffle(index)

        for i in range(k):
            val_i = index[i * fold_size:(i + 1) * fold_size]
            train_i = np.concatenate((index[:i * fold_size], index[(i + 1) * fold_size:]))
            X_train, X_val = X[train_i], X[val_i]
            y_train, y_val = y[train_i], y[val_i]

            if batch_size == 1:
                for j in range(X_train.shape[0]):
                    X_j, y_j = X_train[j:j + 1], y_train[j:j + 1]
                    y_train_pred = model.forward(X_j)
                    train_loss = model.compute_loss(y_j, y_train_pred)
                    epoch_train_loss.append(train_loss)
                    model.backward(y_j, lr=lr)

            else:
                for start in range(0, X_train.shape[0], batch_size):
                    end = start + batch_size
                    X_batch, y_batch = X_train[start:end], y_train[start:end]
                    y_train_pred = model.forward(X_batch)
                    train_loss = model.compute_loss(y_batch, y_train_pred)
                    epoch_train_loss.append(train_loss)
                    model.backward(y_batch, lr=lr)

            y_val_pred = model.forward(X_val)
            val_loss = model.compute_loss(y_val, y_val_pred)
            epoch_val_loss.append(val_loss)

        train_losses.append(np.mean(epoch_train_loss))
        val_losses.append(np.mean(epoch_val_loss))

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss Curves for {task.capitalize()}')
    plt.legend()
    plt.show()

    return model


def nonlinear_data(n_samples=100, noise=0.1):
    X = np.linspace(-2, 2, n_samples).reshape(-1, 1)
    y = np.sin(X * np.pi) + noise * np.random.randn(n_samples, 1)
    y = (y - y.min()) / (y.max() - y.min())
    return X, y


def classification_data(n_samples=100):
    np.random.seed(42)
    X1 = np.random.randn(n_samples // 2, 2) + np.array([1, 1])
    X2 = np.random.randn(n_samples // 2, 2) + np.array([-1, -1])
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).reshape(-1, 1)
    return X, y


if __name__ == "__main__":
    task = input("Choose the task ('r' for function approximation, 'c' for binary classification): ").strip().lower()

    if task == 'r':
        layers = [1, 10, 1]
        X, y = nonlinear_data(n_samples=200, noise=0.1)
        model = train(MLP, layers, X, y, task='regression', k=5, epochs=100, lr=0.5, batch_size=1)

        predictions = model.predict(X)
        plt.scatter(X, y, color='blue', label='True Data')
        plt.plot(X, predictions, color='red', label='Fitted Curve')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('True Data and Fitted Curve for Regression')
        plt.legend()
        plt.show()

    elif task == 'c':

        layers = [2, 10, 1]
        X, y = classification_data(n_samples=200)
        model = train(MLP, layers, X, y, task='classification', k=5, epochs=300, lr=0.01, batch_size=1)

        y_pred = model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        recall = recall_score(y, y_pred)
        precision = precision_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        print("Accuracy:", accuracy)
        print("Recall:", recall)
        print("Precision:", precision)
        print("F1 Score:", f1)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid).reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
        plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolor='k', cmap=plt.cm.Paired, marker='o')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary for Binary Classification')
        plt.show()

    else:
        print("Invalid task. Please choose either 'r' or 'c'.")
