import numpy as np


class SoftKMeans:
    def __init__(self, X, y, k=3, beta=1., max_iter=200):
        self.X = X
        self.y = y
        self.k = k
        self.beta = beta
        self.max_iter = max_iter
        self.centroids = None
        self.r = None

    def initialize(self, seed):
        # 随机初始化质心
        np.random.seed(seed)
        indices = np.random.choice(self.X.shape[0], self.k, replace=False)
        self.centroids = self.X[indices]

    def compute_r(self):
        # 计算每个数据点分配给每个簇的责任值
        d = np.linalg.norm(self.X[:, np.newaxis] - self.centroids, axis=2)
        exp_d = np.exp(-self.beta * d)
        self.r = exp_d / np.sum(exp_d, axis=1, keepdims=True)

    def update_centroids(self):
        # 根据责任值更新质心
        weighted_sum = np.dot(self.r.T, self.X)
        weights = np.sum(self.r, axis=0)[:, np.newaxis]
        self.centroids = weighted_sum / weights

    def fit(self, seed):
        # 执行Soft K-means聚类算法
        self.initialize(seed)
        for _ in range(self.max_iter):
            prev_centroids = self.centroids.copy()
            self.compute_r()
            self.update_centroids()
            if np.allclose(prev_centroids, self.centroids):
                break
        return np.argmax(self.r, axis=1)

    def get_accuracy(self, predicted_clusters):
        from itertools import permutations
        label_permutations = list(permutations(range(self.k)))
        best_accuracy = 0

        for perm in label_permutations:
            mapped_labels = np.array([perm[int(cluster)] for cluster in predicted_clusters]) + 1  # 标签从1开始
            accuracy = np.mean(mapped_labels == self.y)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
        # print(f"Accuracy: {best_accuracy * 100:.2f}%")
        return best_accuracy * 100

    def run(self, seed):
        predicted_clusters = self.fit(seed)
        return self.get_accuracy(predicted_clusters)


if __name__ == "__main__":
    data = np.loadtxt('seeds_dataset.txt')
    X = data[:, :-1]
    y = data[:, -1]
    soft_kmeans = SoftKMeans(X, y, k=3, beta=1)
    total_accuracy = []
    for seed in range(100):
        total_accuracy.append(soft_kmeans.run(seed))
    print(f"Average Accuracy: {np.mean(total_accuracy) :.2f}%  Var: {np.var(total_accuracy) :.2f}")
