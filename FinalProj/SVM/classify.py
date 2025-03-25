import numpy as np
import svc
import importlib

importlib.reload(svc)

file_path = '../seeds_dataset.txt'
data = np.loadtxt(file_path)

filtered_data = data[data[:, -1] != 2]
filtered_data = filtered_data[:140]
X = filtered_data[:, :-1]
y = filtered_data[:, -1]
y = np.where(y == 3, 0, 1)

clf = svc.BiLinearSVC(C=1.,
                      max_iter=1000,
                      tol=1e-5).fit(X, y)

accuracy = clf.score(X, y)
print(f"Linear accuracy: {accuracy:.4f}")

clf = svc.BiKernelSVC(C=1.,
                      kernel='rbf',
                      degree=3,
                      gamma='auto',
                      coef0=0,
                      max_iter=1000,
                      rff=False,
                      D=1000,
                      tol=1e-5).fit(X, y)

accuracy = clf.score(X, y)
print(f"Kernel accuracy: {accuracy:.4f}")
