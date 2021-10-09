import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y = np.array([1.2, 1.5, 1.7, 2, 2.24, 2.4, 2.75, 3])
ω = np.array([1, 1, 50, 1, 1, 1, 1, 1])


# plt.scatter(x, y, marker='o')
# plt.show()

# 解方程
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

#--------------19:20

b2, b1, b0=np.polyfit(x,y,2)

#--------------
#'''
A = gen_coefficient_matrix(x, ω)
b = gen_right_vector(x, y, ω)
print(A)
print(b)

a0, a1, a2 = np.linalg.solve(A, b)
print('a0=', a0, 'a1=', a1, 'a2=', a2)

#'''
# 拟合曲线描点
X = np.arange(0, 1, 0.01)
Y = np.array([a0 + a1 * x + a2 * x ** 2 for x in X])
Y2 = np.array([b0 + b1 * x + b2 * x ** 2 for x in X])
plt.plot(x, y, 'ro', X, Y)
plt.plot(x, y, 'ro', X, Y2)
#plt.title("y = {:.5f} + {:.5f}x + {:.5f}$x^2$ ".format(a0, a1, a2))
plt.show()
