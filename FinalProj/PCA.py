import numpy as np
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        # X_centered = X - self.mean
        X_centered = (X - X.mean(axis=0)) / X.std(axis=0)

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        # X_centered = X - self.mean
        X_centered = (X - X.mean(axis=0)) / X.std(axis=0)
        return np.dot(X_centered, self.components)

    def inverse_transform(self, X_reduced):
        # return np.dot(X_reduced, self.components.T) + self.mean
        return np.dot(X_reduced, self.components.T)

    def reconstruction_error(self, X, X_reconstructed):
        # return np.mean((X - X_reconstructed) ** 2)
        return np.mean(((X - X.mean(axis=0)) / X.std(axis=0) - X_reconstructed) ** 2)


def load_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y


def plot_2d(X, y, title="PCA 2D Projection"):
    plt.figure()
    unique_labels = np.unique(y)
    for label in unique_labels:
        plt.scatter(X[y == label, 0], X[y == label, 1], label=f"Class {label}")
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid()
    plt.show()


def plot_3d(X, y, title="PCA 3D Projection"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(y)
    for label in unique_labels:
        ax.scatter(X[y == label, 0], X[y == label, 1], X[y == label, 2], label=f"Class {label}")
    ax.set_title(title)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    file_path = "seeds_dataset.txt"
    X, y = load_data(file_path)

    # PCA 降维到2维
    pca_2d = PCA(n_components=2)
    pca_2d.fit(X)
    X_2d = pca_2d.transform(X)
    X_2d_reconstructed = pca_2d.inverse_transform(X_2d)
    error_2d = pca_2d.reconstruction_error(X, X_2d_reconstructed)

    print(f"Reconstruction Error for 2D PCA: {error_2d:.6f}")
    plot_2d(X_2d, y, title="PCA with 2 Principal Components")

    # PCA 降维到3维
    pca_3d = PCA(n_components=3)
    pca_3d.fit(X)
    X_3d = pca_3d.transform(X)
    X_3d_reconstructed = pca_3d.inverse_transform(X_3d)
    error_3d = pca_3d.reconstruction_error(X, X_3d_reconstructed)

    print(f"Reconstruction Error for 3D PCA: {error_3d:.6f}")
    plot_3d(X_3d, y, title="PCA with 3 Principal Components")
