#coding:utf-8
import numpy as np
import math
import matplotlib.pyplot as plt


# 读取数据
b = np.loadtxt("ex1data1.txt")

# 定义样本数
m = b.shape[0]

# 初始化输入数据
x = np.ones([m,2],dtype=float)
x[0:m,1:2] = b[0:m,0:1]

# 目标向量
y = b[0:m,1:2]

# 定义Normal Equation
def normalEq(x,y):
	return np.dot(np.dot(np.matrix(np.dot(x.T,x)).I,x.T),y)

# 获得theta
theta = normalEq(x,y)

# 定义预测函数
def h(theta,x_i):
	return x_i*theta

# 定义代价函数
def J(theta,x,y,m):
	result = 0
	for i in range(m):
		diff = h(theta,x[i]) - y[i]
		result = result + pow(diff,2)
	return result/(2*m)

#显示结果

print "the cost is:"
print J(theta,x,y,m)
print "the theta is:"
print theta

z = np.linspace(0,np.max(x[0:m,1:2]),100)
w = theta[0,0] + theta[1,0]*z

plt.figure()
plt.scatter(x[0:m,1:2],y,10,c='y')
plt.plot(z,w)
plt.show()