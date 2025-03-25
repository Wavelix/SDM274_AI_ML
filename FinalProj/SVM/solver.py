import numpy as np
from functools import lru_cache


class Solver:
    r'''SMO算法求解器，迭代求解下面的问题:

    .. math:: \min_{\alpha} \quad & \frac{1}{2} \alpha^T Q \alpha +  p^T \alpha \\
        \text{s.t.} \quad &  y^T \alpha=0 \\
                          & 0 \leq \alpha_i \leq C, i=1,\cdots,l
    
    Parameters
    ----------
    Q : numpy.ndarray
        优化问题中的 :math:`Q` 矩阵；
    p : numpy.ndarray
        优化问题中的 :math:`p` 向量；
    y : numpy.ndarray
        优化问题中的 :math:`y` 向量；
    C : float
        优化问题中的 :math:`C` 变量；
    tol : float, default=1e-5
        变量选择的tolerance，默认为1e-5.
    '''
    def __init__(self,
                 Q: np.ndarray,
                 p: np.ndarray,
                 y: np.ndarray,
                 C: float,
                 tol: float = 1e-5) -> None:
        problem_size = p.shape[0]
        assert problem_size == y.shape[0]
        if Q is not None:
            assert problem_size == Q.shape[0]
            assert problem_size == Q.shape[1]

        self.Q = Q
        self.p = p
        self.y = y
        self.C = C
        self.tol = tol
        self.alpha = np.zeros(problem_size)

        # Calculate -y·▽f(α)
        self.neg_y_grad = -y * p

    def working_set_select(self):
        r'''工作集选择，这里采用一阶选择:

        .. math:: I_{up}(\alpha)  &= \{t|\alpha_t<C, y_t=1 \text{ or } \alpha_t>0, y_t=-1\} \\
                  I_{low}(\alpha) &= \{t|\alpha_t<C, y_t=-1 \text{ or } \alpha_t>0, y_t=1\} \\
                  i & \in \arg\max_{t} \{-y_t \nabla_tf(\alpha) | t \in I_{up}(\alpha)\}  \\
                  j & \in \arg\max_{t} \{-y_t \nabla_tf(\alpha) | t \in I_{low}(\alpha)\} \\
        '''
        Iup = np.argwhere(
            np.logical_or(
                np.logical_and(self.alpha < self.C, self.y > 0),
                np.logical_and(self.alpha > 0, self.y < 0),
            )).flatten()
        Ilow = np.argwhere(
            np.logical_or(
                np.logical_and(self.alpha < self.C, self.y < 0),
                np.logical_and(self.alpha > 0, self.y > 0),
            )).flatten()

        find_fail = False
        try:
            i = Iup[np.argmax(self.neg_y_grad[Iup])]
            j = Ilow[np.argmin(self.neg_y_grad[Ilow])]
        except:
            find_fail = True

        if find_fail or self.neg_y_grad[i] - self.neg_y_grad[j] < self.tol:
            return -1, -1
        return i, j

    def update(self, i: int, j: int, func=None):
        '''变量更新，在保证变量满足约束的条件下对两变量进行更新
        '''
        Qi, Qj = self.get_Q(i, func), self.get_Q(j, func)
        yi, yj = self.y[i], self.y[j]
        alpha_i, alpha_j = self.alpha[i], self.alpha[j]

        quad_coef = Qi[i] + Qj[j] - 2 * yi * yj * Qi[j]
        if quad_coef <= 0:
            quad_coef = 1e-12

        if yi * yj == -1:
            delta = (self.neg_y_grad[i] * yi +
                     self.neg_y_grad[j] * yj) / quad_coef
            diff = alpha_i - alpha_j
            self.alpha[i] += delta
            self.alpha[j] += delta

            if diff > 0:
                if (self.alpha[j] < 0):
                    self.alpha[j] = 0
                    self.alpha[i] = diff

            else:
                if (self.alpha[i] < 0):
                    self.alpha[i] = 0
                    self.alpha[j] = -diff

            if diff > 0:
                if (self.alpha[i] > self.C):
                    self.alpha[i] = self.C
                    self.alpha[j] = self.C - diff

            else:
                if (self.alpha[j] > self.C):
                    self.alpha[j] = self.C
                    self.alpha[i] = self.C + diff

        else:
            delta = (self.neg_y_grad[j] * yj -
                     self.neg_y_grad[i] * yi) / quad_coef
            sum = self.alpha[i] + self.alpha[j]
            self.alpha[i] -= delta
            self.alpha[j] += delta

            if sum > self.C:
                if self.alpha[i] > self.C:
                    self.alpha[i] = self.C
                    self.alpha[j] = sum - self.C

            else:
                if self.alpha[j] < 0:
                    self.alpha[j] = 0
                    self.alpha[i] = sum

            if sum > self.C:
                if self.alpha[j] > self.C:
                    self.alpha[j] = self.C
                    self.alpha[i] = sum - self.C

            else:
                if self.alpha[i] < 0:
                    self.alpha[i] = 0
                    self.alpha[j] = sum

        delta_i = self.alpha[i] - alpha_i
        delta_j = self.alpha[j] - alpha_j
        self.neg_y_grad -= self.y * (delta_i * Qi + delta_j * Qj)
        return delta_i, delta_j

    def calculate_rho(self) -> float:
        r'''计算偏置项
        
        .. math:: \rho = \frac{\sum_{i:0<\alpha_i<C} y_i \nabla_if(\alpha)}{|\{i: 0<\alpha_i<C\}|}

        如果不存在满足条件的支持向量，那么

        .. math:: -M(\alpha) &=\max\{y_i\nabla_if(\alpha) | \alpha_i=0, y_i=-1 \text{ or } \alpha_i=C, y_i=1\} \\
                  -m(\alpha) &= \max\{y_i\nabla_if(\alpha)| \alpha_i=0, y_i=1  \text{ or } \alpha_i=C, y_i=-1\} \\
                  \rho &= -\dfrac{M(\alpha)+m(\alpha)}{2}
        '''
        sv = np.logical_and(
            self.alpha > 0,
            self.alpha < self.C,
        )
        if sv.sum() > 0:
            rho = -np.average(self.neg_y_grad[sv])
        else:
            ub_id = np.logical_or(
                np.logical_and(self.alpha == 0, self.y < 0),
                np.logical_and(self.alpha == self.C, self.y > 0),
            )
            lb_id = np.logical_or(
                np.logical_and(self.alpha == 0, self.y > 0),
                np.logical_and(self.alpha == self.C, self.y < 0),
            )
            rho = -(self.neg_y_grad[lb_id].min() +
                    self.neg_y_grad[ub_id].max()) / 2
        return rho

    def get_Q(self, i: int, func=None):
        '''获取核矩阵的第i行/列，即
        
        .. math:: [K(x_1, x_i),\cdots, K(x_l, x_i)]
        '''
        return self.Q[i]

