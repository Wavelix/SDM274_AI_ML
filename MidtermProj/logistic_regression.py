import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, lr=0.01, decay=0.01, epochs=1000, method="BGD", batch_size=32, class_W=None):

        self.lr = lr
        self.decay = decay
        self.epochs = epochs
        self.method = method
        self.batch_size = batch_size
        self.class_W = class_W
        self.W = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

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

    def get_class_W(self, y):
        unique, counts = np.unique(y, return_counts=True)
        total = sum(counts)
        return {cls: total / count for cls, count in zip(unique, counts)}

    def train(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.W = np.zeros((n_features, 1))
        self.b = 0
        losses = []

        if self.class_W is None:
            self.class_W = self.get_class_W(y_train)

        for epoch in range(self.epochs):

            current_lr = self.lr / (1 + self.decay * epoch)

            if self.method == "BGD":
                linear_model = np.dot(X_train, self.W) + self.b
                y_pred = self.sigmoid(linear_model)

                weights = np.where(y_train == 1, self.class_W[1], self.class_W[0])
                loss = -np.mean(
                    weights * (y_train * np.log(y_pred + 1e-15) + (1 - y_train) * np.log(1 - y_pred + 1e-15)))
                losses.append(loss)

                dz = weights * (y_pred - y_train)
                dw = np.dot(X_train.T, dz) / n_samples
                db = np.sum(dz) / n_samples

                self.W -= current_lr * dw
                self.b -= current_lr * db

            elif self.method == "MBGD":
                indices = np.random.permutation(n_samples)
                X_train_shuffled = X_train[indices]
                y_train_shuffled = y_train[indices]

                for start_idx in range(0, n_samples, self.batch_size):
                    end_idx = start_idx + self.batch_size
                    X_batch = X_train_shuffled[start_idx:end_idx]
                    y_batch = y_train_shuffled[start_idx:end_idx]

                    linear_model = np.dot(X_batch, self.W) + self.b
                    y_pred = self.sigmoid(linear_model)

                    weights = np.where(y_batch == 1, self.class_W[1], self.class_W[0])
                    dz = weights * (y_pred - y_batch)
                    dw = np.dot(X_batch.T, dz) / len(X_batch)
                    db = np.sum(dz) / len(X_batch)

                    self.W -= current_lr * dw
                    self.b -= current_lr * db

                linear_model_all = np.dot(X_train, self.W) + self.b
                y_pred_all = self.sigmoid(linear_model_all)
                W_all = np.where(y_train == 1, self.class_W[1], self.class_W[0])
                loss = -np.mean(
                    W_all * (y_train * np.log(y_pred_all + 1e-15) + (1 - y_train) * np.log(1 - y_pred_all + 1e-15)))
                losses.append(loss)

        plt.figure()
        plt.plot(range(self.epochs), losses, label="Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.show()

    def predict(self, X):
        linear_model = np.dot(X, self.W) + self.b
        y_pred = self.sigmoid(linear_model)
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
    lr_model = LogisticRegression(lr=0.01, decay=0.01, epochs=200, method="MBGD", batch_size=64)
    X_train, X_test, y_train, y_test = lr_model.data_process(file_path)
    lr_model.train(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    lr_model.evaluate(y_test, y_pred)
