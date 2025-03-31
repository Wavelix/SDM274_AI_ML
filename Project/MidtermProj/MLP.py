import numpy as np
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, layers, activations, lr=0.01, epochs=1000, batch_size=32, class_W=None):

        self.layers = layers
        self.activations = activations
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_W = class_W
        self.W = []
        self.b = []
        self.init_W()

    def init_W(self):
        np.random.seed(1)
        for i in range(len(self.layers) - 1):
            W = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i])
            b = np.zeros((1, self.layers[i + 1]))
            self.W.append(W)
            self.b.append(b)

    def activation(self, x, func):
        if func == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif func == "relu":
            return np.maximum(0, x)
        elif func == "tanh":
            return np.tanh(x)

    def d_activation(self, x, func):
        if func == "sigmoid":
            s = self.activation(x, "sigmoid")
            return s * (1 - s)
        elif func == "relu":
            return np.where(x > 0, 1, 0)
        elif func == "tanh":
            return 1 - np.tanh(x) ** 2

    def data_process(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        data = [line.strip().split(',') for line in lines[1:]]
        data = np.array(data)

        type = data[:, 2]
        type_to_num = {'M': 0, 'L': 1, 'H': 2}
        type_numeric = np.array([type_to_num[typ] for typ in type])

        features = data[:, 3:8].astype(float)
        target = data[:, 8].astype(int)

        X = np.column_stack((type_numeric, features))
        y = target

        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        oversampled_pos_indices = np.random.choice(pos_indices, size=len(neg_indices), replace=True)
        oversampled_indices = np.concatenate([neg_indices, oversampled_pos_indices])
        np.random.shuffle(oversampled_indices)

        X = X[oversampled_indices]
        y = y[oversampled_indices]

        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Standardize features
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0)
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

        return X_train, X_test, y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    def forward(self, X):
        self.z = []
        self.a = [X]
        for i in range(len(self.W)):
            z = np.dot(self.a[-1], self.W[i]) + self.b[i]
            self.z.append(z)
            if i == len(self.W) - 1:
                a = self.activation(z, "sigmoid")
            else:
                a = self.activation(z, self.activations[i])
            self.a.append(a)
        return self.a[-1]

    def backward(self, y_batch, class_W):
        m = y_batch.shape[0]
        dz = self.a[-1] - y_batch
        dw = np.dot(self.a[-2].T, dz * class_W) / m
        db = np.sum(dz * class_W, axis=0, keepdims=True) / m
        self.dweights = [dw]
        self.dbiases = [db]

        for i in range(len(self.W) - 2, -1, -1):
            dz = np.dot(dz, self.W[i + 1].T) * self.d_activation(self.z[i], self.activations[i])
            dw = np.dot(self.a[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            self.dweights.insert(0, dw)
            self.dbiases.insert(0, db)

    def update_W(self):
        for i in range(len(self.W)):
            self.W[i] -= self.lr * self.dweights[i]
            self.b[i] -= self.lr * self.dbiases[i]

    def train(self, X_train, y_train):
        losses = []
        n_samples = X_train.shape[0]

        if self.class_W is None:
            unique, counts = np.unique(y_train, return_counts=True)
            self.class_W = {cls: 1 / count for cls, count in zip(unique, counts)}

        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                class_W = np.where(y_batch == 1, self.class_W[1], self.class_W[0]).reshape(-1, 1)
                self.forward(X_batch)
                self.backward(y_batch, class_W)
                self.update_W()

            y_pred = self.forward(X_train)
            loss = -np.mean(y_train * np.log(y_pred + 1e-15) + (1 - y_train) * np.log(1 - y_pred + 1e-15))
            losses.append(loss)

        plt.plot(range(self.epochs), losses, label="Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred >= 0.5).astype(int)

    def evaluate(self, y_true, y_pred):
        accuracy = np.mean(y_true == y_pred)
        precision = np.sum((y_true == 1) & (y_pred == 1)) / (np.sum(y_pred == 1) + 1e-15)
        recall = np.sum((y_true == 1) & (y_pred == 1)) / (np.sum(y_true == 1) + 1e-15)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-15)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")


if __name__ == "__main__":
    file_path = 'ai4i2020.csv'
    mlp_model = MLP(layers=[6, 8,8, 1], activations=["relu",'relu'], lr=0.01, epochs=1000, batch_size=64)
    X_train, X_test, y_train, y_test = mlp_model.data_process(file_path)
    mlp_model.train(X_train, y_train)
    y_pred = mlp_model.predict(X_test)
    mlp_model.evaluate(y_test, y_pred)
