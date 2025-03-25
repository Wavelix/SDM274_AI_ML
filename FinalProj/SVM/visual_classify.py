from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
import svc
import importlib
importlib.reload(svc)

from PCA import PCA, load_data as load_pca_data

try:
    import seaborn as sns
    sns.set()
except:
    pass

RANDOM_STATE = 202

file_path = '../seeds_dataset.txt'
data = np.loadtxt(file_path)
filtered_data = data[data[:, -1] != 2]
filtered_data = filtered_data[:140]
X_ = filtered_data[:, :-1]  # 特征
y = filtered_data[:, -1]  # 标签
y=np.where(y==3, 0, 1)
pca_2d = PCA(n_components=2)
pca_2d.fit(X_)
X = pca_2d.transform(X_)


x_min = np.min(X[:, 0])
x_max = np.max(X[:, 0])
y_min = np.min(X[:, 1])
y_max = np.max(X[:, 1])

plot_x = np.linspace(x_min - 1, x_max + 1, 1001)
plot_y = np.linspace(x_min - 1, x_max + 1, 1001)
xx, yy = np.meshgrid(plot_x, plot_y)

clf = svc.BiLinearSVC(C = 1.,
                 max_iter = 1000,
                 tol = 1e-5).fit(X, y)
accuracy = clf.score(X, y)
print(f"Linear accuracy: {accuracy:.4f}")
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap = plt.cm.coolwarm

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.contourf(xx, yy, Z, cmap=cmap)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('SVC with linear kernel(Decision function view)')

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.subplot(2, 2, 2)
plt.contourf(xx, yy, Z, cmap=cmap)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('SVC with linear kernel(0-1 view)')

clf = svc.BiKernelSVC(C = 1.,
                 kernel = 'rbf',
                 degree = 3,
                 gamma = 'scale',
                 coef0 = 0,
                 max_iter = 1000,
                 rff = False,
                 D = 1000,
                 tol = 1e-5).fit(X, y)
accuracy = clf.score(X, y)
print(f"Kernel accuracy: {accuracy:.4f}")
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(2, 2, 3)
plt.contourf(xx, yy, Z, cmap=cmap)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('SVC with Gaussian kernel(Decision function view)')

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.subplot(2, 2, 4)
plt.contourf(xx, yy, Z, cmap=cmap)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('SVC with Gaussian kernel(0-1 view)')

plt.show()
print(f'X.shape={X.shape}; y.shape={y.shape}')