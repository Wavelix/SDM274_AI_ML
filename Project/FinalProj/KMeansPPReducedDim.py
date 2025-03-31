import numpy as np
from PCA import PCA, load_data as load_pca_data
from AE import Autoencoder, load_data as load_ae_data
from KMeansPP import KMeansPP

if __name__ == "__main__":
    # 从PCA降维数据中读取
    file_path = "seeds_dataset.txt"
    X, y = load_pca_data(file_path)
    # PCA 2维数据
    pca_2d = PCA(n_components=2)
    pca_2d.fit(X)
    X_2d = pca_2d.transform(X)
    kmeans_pca_2d = KMeansPP(X_2d, y, k=3)
    total_accuracy = []
    for seed in range(100):
        total_accuracy.append(kmeans_pca_2d.run(seed))
    print(f"PCA 2D Average Accuracy: {np.mean(total_accuracy) :.2f}%  Var: {np.var(total_accuracy) :.2f}")
    # PCA 3维数据
    pca_3d = PCA(n_components=3)
    pca_3d.fit(X)
    X_3d = pca_3d.transform(X)
    kmeans_pca_3d = KMeansPP(X_3d, y, k=3)
    total_accuracy = []
    for seed in range(100):
        total_accuracy.append(kmeans_pca_3d.run(seed))
    print(f"PCA 3D Average Accuracy: {np.mean(total_accuracy) :.2f}%  Var: {np.var(total_accuracy) :.2f}")

    # 从AE降维数据中读取
    X, y = load_ae_data(file_path)
    X = X - X.mean(axis=0)
    # AE 2维数据
    ae_2d = Autoencoder(input_dim=X.shape[1], hidden_layer=[2], lr=0.04, epochs=200000)
    ae_2d.train(X)
    X_encoded_2d = ae_2d.encode(X)
    kmeans_ae_2d = KMeansPP(X_encoded_2d, y, k=3)
    total_accuracy = []
    for seed in range(100):
        total_accuracy.append(kmeans_ae_2d.run(seed))
    print(f"AE  2D Average Accuracy: {np.mean(total_accuracy) :.2f}%  Var: {np.var(total_accuracy) :.2f}")
    # AE 3维数据
    ae_3d = Autoencoder(input_dim=X.shape[1], hidden_layer=[3], lr=0.02, epochs=200000)
    ae_3d.train(X)
    X_encoded_3d = ae_3d.encode(X)
    kmeans_ae_3d = KMeansPP(X_encoded_3d, y, k=3)
    total_accuracy = []
    for seed in range(100):
        total_accuracy.append(kmeans_ae_3d.run(seed))
    print(f"AE  3D Average Accuracy: {np.mean(total_accuracy) :.2f}%  Var: {np.var(total_accuracy) :.2f}")
