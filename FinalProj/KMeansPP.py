import numpy as np


class KMeansPP:

    def __init__(self, X, y, k=3, max_iter=200):
        self.X = X
        self.y = y
        self.k = k
        self.max_iter = max_iter
        self.centroids = None

    def initialize(self, seed):
        np.random.seed(seed)
        centroids = [self.X[np.random.randint(0, self.X.shape[0])]]
        for _ in range(1, self.k):
            d = np.array([min(np.linalg.norm(point - c) ** 2 for c in centroids) for point in self.X])
            P = d / d.sum()
            cumulative_probabilities = P.cumsum()
            r = np.random.rand()
            for i, p in enumerate(cumulative_probabilities):
                if r < p:
                    centroids.append(self.X[i])
                    break
        self.centroids = np.array(centroids)

    def assign(self):
        # 将样本分配到最近的质心
        d = np.linalg.norm(self.X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(d, axis=1)

    def update_centroids(self, cluster_assignments):
        # 根据分配的簇更新质心
        new_centroids = np.array([self.X[cluster_assignments == i].mean(axis=0) for i in range(self.k)])
        return new_centroids

    def fit(self, seed):
        # K-means++
        self.initialize(seed)
        cluster_assignments = None
        for _ in range(self.max_iter):
            cluster_assignments = self.assign()
            new_centroids = self.update_centroids(cluster_assignments)
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return cluster_assignments

    def get_accuracy(self, predicted_clusters):
        from itertools import permutations
        label_permutations = list(permutations(range(self.k)))
        best_accuracy = 0

        for perm in label_permutations:
            mapped_labels = np.array([perm[int(cluster)] for cluster in predicted_clusters]) + 1  # 标签从1开始
            accuracy = np.mean(mapped_labels == self.y)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
        return best_accuracy * 100


    def run(self, seed):
        predicted_clusters = self.fit(seed)
        return self.get_accuracy(predicted_clusters)


if __name__ == "__main__":

    data = np.loadtxt('seeds_dataset.txt')
    X = data[:, :-1]
    y = data[:, -1]
    kmeans = KMeansPP(X, y, k=3)

    total_accuracy = []
    for seed in range(100):
        total_accuracy.append(kmeans.run(seed))
    print(f"Average Accuracy: {np.mean(total_accuracy) :.2f}%  Var: {np.var(total_accuracy) :.2f}")
