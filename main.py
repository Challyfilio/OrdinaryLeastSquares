import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y = np.array([1.2, 1.5, 1.7, 2, 2.24, 2.4, 2.75, 3])
ω = np.array([1, 1, 50, 1, 1, 1, 1, 1])
plt.scatter(x, y, marker='o')
plt.show()


# 生成系数矩阵A
def gen_coefficient_matrix(X, Y):
    N = len(X)
    m = 3
    A = []
    # 计算每一个方程的系数
    for i in range(m):
        a = []
        # 计算当前方程中的每一个系数
        for j in range(m):
            a.append(sum(X ** (i + j)))
        A.append(a)
    return A


# #
# 计算方程组的右端向量b
def gen_right_vector(X, Y):
    N = len(X)
    m = 3
    b = []
    for i in range(m):
        b.append(sum(X ** i * Y))
    return b


A = gen_coefficient_matrix(x, y)
b = gen_right_vector(x, y)
# #
a0, a1, a2 = np.linalg.solve(A, b)
#
# 生成拟合曲线的绘制点
_X = np.arange(0, 1, 0.01)
_Y = np.array([a0 + a1 * x + a2 * x ** 2 for x in _X])

plt.plot(x, y, 'ro', _X, _Y, 'b', linewidth=2)
plt.title("y = {} + {}x + {}$x^2$ ".format(a0, a1, a2))
plt.axis([-0.1, 1.1, 1.1, 3.1])
plt.show()
