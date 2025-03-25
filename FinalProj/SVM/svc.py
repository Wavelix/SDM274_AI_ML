import numpy as np 
from sklearn.metrics import accuracy_score 
from rff import NormalRFF
from solver import Solver


class BiLinearSVC:
    r'''二分类线性SVM
    
    通过求解对偶问题

    .. math:: \min_{\alpha} \quad & \frac{1}{2} \alpha^T Q \alpha + p^T \alpha \\
                \text{s.t.} \quad & y^T \alpha = 0, \\
                                  & 0 \leq \alpha_i \leq C, i=1,\cdots,N

    得到决策边界

    .. math:: f(x) = \sum_{i=1}^N y_i \alpha_i  x_i^T x - \rho

    Parameters
    ----------
    C : float, default=1
        SVM的正则化参数，默认为1；
    max_iter : int, default=1000
        SMO算法迭代次数，默认1000；
    tol : float, default=1e-5
        SMO算法的容忍度参数，默认1e-5.
    '''
    def __init__(self,
                 C: float = 1.,
                 max_iter: int = 1000,
                 tol: float = 1e-5) -> None:
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.tol = tol 

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''训练模型

        Parameters
        ----------
        X : np.ndarray
            训练集特征;
        y : np.array
            训练集标签，建议0为负标签，1为正标签.
        '''
        X, y = np.array(X), np.array(y, dtype=float)
        y[y != 1] = -1
        N, self.n_features = X.shape
        p = -np.ones(N)

        w = np.zeros(self.n_features)
        Q = y.reshape(-1, 1) * y * np.matmul(X, X.T)
        solver = Solver(Q, p, y, self.C, self.tol)
        
        def func(i):
            return y * np.matmul(X, X[i]) * y[i]

        for n_iter in range(self.max_iter):
            i, j = solver.working_set_select()
            if i < 0:
                break

            delta_i, delta_j = solver.update(i, j, func)
            w += delta_i * y[i] * X[i] + delta_j * y[j] * X[j]
        else:
            print("LinearSVC not coverage with {} iterations".format(
                self.max_iter))

        self.coef_ = (w, solver.calculate_rho())
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        '''决策函数，输出预测值'''
        return np.matmul(self.coef_[0], np.array(X).T) - self.coef_[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''预测函数，输出预测标签(0-1)'''
        return (self.decision_function(np.array(X)) >= 0).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        '''评估函数，给定特征和标签，输出正确率'''
        return accuracy_score(y, self.predict(X))

 
class BiKernelSVC(BiLinearSVC):
    r'''二分类核SVM,优化问题与BiLinearSVC相同,只是Q矩阵定义不同。

    此时的决策边界

    .. math:: f(x) = \sum_{i=1}^N y_i \alpha_i K(x_i, x) - \rho

    Parameters
    ----------
    C : float, default=1
        SVM的正则化参数，默认为1；
    kernel : {"linear", "poly", "rbf", "sigmoid"}, default="rbf"
        核函数，默认径向基函数(RBF)；
    degree : float, default=3
        多项式核的次数，默认3；
    gamma : {"scale", "auto", float}, default="scale"
        rbf、ploy和sigmoid核的参数 :math:`\gamma`，如果用'scale'，那么就是1 / (n_features * X.var())，如果用'auto'，那么就是1 / n_features；
    coef0 : float, default=0.
        核函数中的独立项。它只在"poly"和"sigmoid"中有意义；
    max_iter : int, default=1000
        SMO算法迭代次数，默认1000；
    rff : bool, default=False
        是否采用随机傅里叶特征，默认为False；
    D : int, default=1000
        随机傅里叶特征的采样次数，默认为1000；
    tol : float, default=1e-5
        SMO算法的容忍度参数，默认1e-5.
    '''
    def __init__(self,
                 C: float = 1.,
                 kernel: str = 'rbf',
                 degree: float = 3,
                 gamma: str = 'scale',
                 coef0: float = 0,
                 max_iter: int = 1000,
                 rff: bool = False,
                 D: int = 1000,
                 tol: float = 1e-5) -> None:
        super().__init__(C, max_iter, tol)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.rff = rff
        self.D = D

    def register_kernel(self, std: float):
        '''注册核函数
        
        Parameters
        ----------
        std : 输入数据的标准差，用于rbf='scale'的情况
        '''
        if type(self.gamma) == str:
            gamma = {
                'scale': 1 / (self.n_features * std),
                'auto': 1 / self.n_features,
            }[self.gamma]
        else:
            gamma = self.gamma

        if self.rff:
            rff = NormalRFF(gamma, self.D).fit(np.ones((1, self.n_features)))
            rbf_func = lambda x, y: np.matmul(rff.transform(x),
                                              rff.transform(y).T)
        else:
            rbf_func = lambda x, y: np.exp(-gamma * (
                (x**2).sum(1, keepdims=True) +
                (y**2).sum(1) - 2 * np.matmul(x, y.T)))

        degree = self.degree
        coef0 = self.coef0
        return {
            "linear": lambda x, y: np.matmul(x, y.T),
            "poly": lambda x, y: (gamma * np.matmul(x, y.T) + coef0)**degree,
            "rbf": rbf_func,
            "sigmoid": lambda x, y: np.tanh(gamma * np.matmul(x, y.T) + coef0)
        }[self.kernel]

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = np.array(X), np.array(y, dtype=float)
        y[y != 1] = -1
        N, self.n_features = X.shape
        p = -np.ones(N)

        kernel_func = self.register_kernel(X.std())

        Q = y.reshape(-1, 1) * y * kernel_func(X, X)
        solver = Solver(Q, p, y, self.C, self.tol)
        

        def func(i):
            return y * kernel_func(X, X[i:i + 1]).flatten() * y[i]

        for n_iter in range(self.max_iter):
            i, j = solver.working_set_select()
            if i < 0:
                break
            solver.update(i, j, func)
        else:
            print("KernelSVC not coverage with {} iterations".format(
                self.max_iter))

        self.decision_function = lambda x: np.matmul(
            solver.alpha * y,
            kernel_func(X, x),
        ) - solver.calculate_rho()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return super().predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return super().score(X, y)

