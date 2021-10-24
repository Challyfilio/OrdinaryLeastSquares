import numpy as np
import matplotlib.pyplot as plt

# data
x = np.array([0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y = np.array([1.2, 1.5, 1.7, 2.0, 2.24, 2.4, 2.75, 3.0])
ω = np.array([1, 1, 50, 1, 1, 1, 1, 1])
ω0 = np.array([1, 1, 1, 1, 1, 1, 1, 1])


# 解方程 AX=b
# 系数矩阵A
def gen_coefficient_matrix(X, omega):
    m = 3
    A = []
    # 计算每一个方程的系数
    for i in range(m):
        a = []
        # 计算当前方程中的每一个系数
        for j in range(m):
            a.append(sum(omega * (X ** (i + j))))  # ω*x^(i+j)
        A.append(a)
    return A


# 计算方程组的右端向量b
def gen_right_vector(X, Y, omega):
    m = 3
    b = []
    for i in range(m):
        b.append(sum(omega * (X ** i * Y)))  # ω*x^i*y
    return b


A = gen_coefficient_matrix(x, ω)
b = gen_right_vector(x, y, ω)
C = gen_coefficient_matrix(x, ω0)
c = gen_right_vector(x, y, ω0)
# print(A)
# print(b)

a0, a1, a2 = np.linalg.solve(A, b)
b0, b1, b2 = np.linalg.solve(C, c)
# print(a0, a1, a2)
# print(b0, b1, b2)

# 画图
plt.scatter(x, y, color='r')
fit_X = np.arange(0, 1, 0.001)
fit_Y1 = np.array([a0 + a1 * x + a2 * x ** 2 for x in fit_X])
fit_Y1b = np.array([b0 + b1 * x + b2 * x ** 2 for x in fit_X])
plt.plot(fit_X, fit_Y1, label='omega(3)=50')
plt.plot(fit_X, fit_Y1b, label='omega(3)=1')
print("S(x) = {:.5f}x^2 + {:.5f}x + {:.5f} ".format(a2, a1, a0))
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# 正交多项式
# 计算alpha
def cal_alpha(x, p, i, omega):
    temp1 = np.dot(x * p[i - 1], p[i - 1] * omega)
    temp2 = np.dot(p[i - 1], p[i - 1] * omega)
    return temp1 / temp2


# 计算beta
def cal_beta(p, i, omega):
    temp1 = np.dot(p[i - 1], p[i - 1] * omega)
    temp2 = np.dot(p[i - 2], p[i - 2] * omega)
    return temp1 / temp2


# 计算P(x)值
def P(X, alpha, beta, i):
    if i == 0:
        return 1  # p0=1
    if i == 1:
        return X - alpha[i]  # p1=x-α
    else:
        return (X - alpha[i]) * P(X, alpha, beta, i - 1) - beta[i - 1] * P(X, alpha, beta, i - 2)


# 正交多项式计算
def Orthogonal(x, y, omega, n, fit_x):
    a, p = [], []
    p.append(np.ones(shape=x.shape))  # p0=1
    alpha, beta = [0], [0]
    for i in range(1, n + 1):
        a.append(np.dot(p[i - 1], y * omega) / np.dot(p[i - 1], p[i - 1] * omega))
        alpha.append(cal_alpha(x, p, i, omega))
        if i == 1:
            p.append(x - alpha[i])
        else:
            beta.append(cal_beta(p, i, omega))
            p.append((x - alpha[i]) * p[i - 1] - beta[i - 1] * p[i - 2])
        if i == n:
            a.append(np.dot(p[n], y * omega) / np.dot(p[n], p[n] * omega))
    # 计算函数值
    fit_y = []
    for j in range(len(fit_X)):
        res = 0
        for k in range(n + 1):
            res += a[k] * P(fit_x[j], alpha, beta, k)
        fit_y.append(res)
    return fit_y


# 画图
plt.scatter(x, y, color='r')
for m in range(2, 7):  # m次多项式拟合
    fit_Y2 = Orthogonal(x, y, ω, m, fit_X)
    plt.plot(fit_X, fit_Y2, label='m=' + str(m))
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
