import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, lr=0.01, epochs=1000, batch_size=None):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.W = None
        self.b = None
        self.class_W = None

    def data_process(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        header = lines[0].strip().split(',')
        data = [line.strip().split(',') for line in lines[1:]]
        data = np.array(data)

        type = data[:, 2]
        type_to_num = {'M': 0, 'L': 1, 'H': 2}
        type_numeric = np.array([type_to_num[typ] for typ in type])

        features = data[:, 3:8].astype(float)
        target = data[:, 8].astype(int)
        target[target == 0] = -1

        X = np.column_stack((type_numeric, features))
        y = target

        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == -1)[0]
        oversampled_pos_indices = np.random.choice(pos_indices, size=len(neg_indices), replace=True)
        oversampled_indices = np.concatenate([neg_indices, oversampled_pos_indices])
        np.random.shuffle(oversampled_indices)

        X = X[oversampled_indices]
        y = y[oversampled_indices]

        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # X_mean = X_train.mean(axis=0)
        # X_std = X_train.std(axis=0)
        # X_train = (X_train - X_mean) / X_std
        # X_test = (X_test - X_mean) / X_std

        return X_train, X_test, y_train, y_test

    def get_class_W(self, y):
        unique, counts = np.unique(y, return_counts=True)
        total = sum(counts)
        weights = {cls: total / count for cls, count in zip(unique, counts)}
        return weights

    def train(self, X_train, y_train, use_mbgd=False):
        n_samples, n_features = X_train.shape
        # self.W = np.zeros(n_features)
        self.W = np.random.randn(n_features) * 0.01
        self.b = 0
        self.class_W = self.get_class_W(y_train)

        losses = []

        for epoch in range(self.epochs):
            total_loss = 0
            lr = self.lr / (1 + 0.01 * epoch)

            if use_mbgd and self.batch_size:
                indices = np.arange(n_samples)
                np.random.shuffle(indices)

                for start in range(0, n_samples, self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    X_batch = X_train[batch_indices]
                    y_batch = y_train[batch_indices]

                    for i in range(len(y_batch)):
                        weight = self.class_W[y_batch[i]]
                        if y_batch[i] * (np.dot(X_batch[i], self.W) + self.b) <= 0:
                            total_loss += 1
                            self.W += lr * weight * y_batch[i] * X_batch[i]
                            self.b += lr * weight * y_batch[i]
            else:
                for i in range(n_samples):
                    weight = self.class_W[y_train[i]]
                    if y_train[i] * (np.dot(X_train[i], self.W) + self.b) <= 0:
                        total_loss += 1
                        self.W += lr * weight * y_train[i] * X_train[i]
                        self.b += lr * weight * y_train[i]

            losses.append(total_loss/n_samples)

        plt.figure()
        plt.plot(range(self.epochs), losses, label="Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.show()

    def predict(self, X):
        y_pred = np.dot(X, self.W) + self.b
        return np.sign(y_pred)

    def evaluate(self, y_true, y_pred):
        y_true_binary = (y_true == 1).astype(int)
        y_pred_binary = (y_pred == 1).astype(int)

        accuracy = np.mean(y_true_binary == y_pred_binary)
        precision = np.sum((y_true_binary == 1) & (y_pred_binary == 1)) / (np.sum(y_pred_binary == 1) + 1e-15)
        recall = np.sum((y_true_binary == 1) & (y_pred_binary == 1)) / (np.sum(y_true_binary == 1) + 1e-15)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-15)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")


if __name__ == "__main__":
    file_path = 'ai4i2020.csv'
    perceptron_model = Perceptron(lr=1e-2, epochs=1000, batch_size=64)
    X_train, X_test, y_train, y_test = perceptron_model.data_process(file_path)
    perceptron_model.train(X_train, y_train, use_mbgd=True)
    y_pred = perceptron_model.predict(X_test)
    perceptron_model.evaluate(y_test, y_pred)
