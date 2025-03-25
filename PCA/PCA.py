import numpy as np
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, file_path):
        self.data = self.load_data(file_path)
        self.labels = self.data[:, 0]
        self.X = self.data[:, 1:]

    def load_data(self, file_path):
        return np.genfromtxt(file_path, delimiter=',')

    def normalization(self):
        self.mean = np.mean(self.X, axis=0)
        self.std = np.std(self.X, axis=0)
        self.X_normalized = (self.X - self.mean) / self.std
        # self.range=np.max(self.X,axis=0)-np.min(self.X,axis=0)
        # self.X_normalized= (self.X-np.min(self.X,axis=0))/(self.range)

    def compute_pca(self):
        cov_matrix = np.cov(self.X_normalized, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        indices = np.argsort(eigenvalues)[::-1]
        self.top_eigenvectors = eigenvectors[:, indices[:2]]

    def project(self):
        self.reduced_data = np.dot(self.X_normalized, self.top_eigenvectors)

    def reconstruct(self):
        # self.reconstructed_data = np.dot(self.reduced_data, self.top_eigenvectors.T) * self.std + self.mean
        self.reconstructed_data = np.dot(self.reduced_data, self.top_eigenvectors.T)
        reconstruction_error = np.mean((self.X_normalized - self.reconstructed_data) ** 2)
        print(f"Reconstruction Error: {reconstruction_error:.6f}")

    def visualize(self):
        for label in np.unique(self.labels):
            indices = np.where(self.labels == label)
            plt.scatter(
                self.reduced_data[indices, 0],
                self.reduced_data[indices, 1],
                label=f"Class {int(label)}"
            )
        plt.title("PCA")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid()
        plt.show()

    def run(self):
        self.normalization()
        self.compute_pca()
        self.project()
        self.reconstruct()
        self.visualize()


pca = PCA('wine.data')
pca.run()
