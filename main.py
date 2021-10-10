import numpy as np
import matplotlib.pyplot as plt
import sympy

# data
x = np.array([0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y = np.array([1.2, 1.5, 1.7, 2, 2.24, 2.4, 2.75, 3])
ω = np.array([1, 1, 50, 1, 1, 1, 1, 1])


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
# print(A)
# print(b)

a0, a1, a2 = np.linalg.solve(A, b)
print(a0, a1, a2)


# 正交多项式
def MathFunc(x, y, omega):
    gram_list_all = []
    d_list = []

    for i in range(2 * 2 + 1):
        gram_list_each = 0
        for x_i, w_i in zip(x, omega):
            gram_list_each += w_i * (x_i ** i)
        gram_list_all.append(gram_list_each)
    for i in range(3):
        t = 0
        for x_i, y_i, w_i in zip(x, y, omega):
            t += w_i * (x_i ** i) * y_i
        d_list.append(t)
    return gram_list_all, d_list


g, d = MathFunc(x, y, ω)
# print(g)
# print(d)

cal0 = sympy.symbols('0')
cal1 = sympy.symbols('1')
cal2 = sympy.symbols('2')

# 正交多项式族
w = [g[0] * cal0 + g[1] * cal1 + g[2] * cal2 - d[0],
     g[1] * cal0 + g[2] * cal1 + g[3] * cal2 - d[1],
     g[2] * cal0 + g[3] * cal1 + g[4] * cal2 - d[2]]

res = sympy.solve(w, [cal0, cal1, cal2])
b0, b1, b2 = res[cal0], res[cal1], res[cal2]
print(b0, b1, b2)

# 散点
plt.scatter(x, y, color='r')

# 拟合曲线画图
X = np.arange(0, 1, 0.01)

# Y1 = np.array([a0 + a1 * x + a2 * x ** 2 for x in X])
# plt.plot(X, Y1, color='b')
# plt.title("(a) y = {:.5f}$x^2$ + {:.5f}x + {:.5f} ".format(a2, a1, a0))

Y2 = np.array([b0 + b1 * x + b2 * x ** 2 for x in X])
plt.plot(X, Y2, color='g')
plt.title("(b) y = {:.5f}$x^2$ + {:.5f}x + {:.5f} ".format(b2, b1, b0))
plt.show()
