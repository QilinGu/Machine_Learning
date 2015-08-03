import os
#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
b = np.loadtxt("ex1data1.txt")

# 初始化输入数据
x = np.ones((b.shape[0],2),dtype=float)
x[0:x.shape[0],1:2] = b[0:b.shape[0],0:1]

# 目标数据
y = b[0:b.shape[0],1:2]

# 定义theta向量
theta = np.matrix([[0.0],[0.0]])

# 样本数m
m = x.shape[0]

# 特征数
n = x.shape[1]

# 特征的最大值
x_max = np.max(x[0:m,1:2])


# 定义最大迭代次数
max_loop = 10000

# 定义收敛精度
eplison = 0.001

# 定义学习率
rate = 0.01

# 定义预测函数
def h(theta,x_i):
	return x_i*theta

# 定义代价函数
def J(theta,rate,x,y,m):
	result = 0
	for i in range(m):
		diff = h(theta,x[i]) - y[i]
		result = result + pow(diff,2)
	return result/(2*m)

# 定义随机梯度下降函数
def sgd(theta,rate,x,y,m,n):
	count = 0
	error = np.matrix([[0.0],[0.0]])
	while count <= max_loop:
		count = count+1
		for i in range(m):
			diff = y[i] - h(theta,x[i])
			for j in range(n):
				theta[j] = theta[j] + rate*diff*x[i,j]
		if( np.max(abs(theta-error)) < eplison ):
			break;
		else:
			error = theta.copy()
	print "Iteration has been exuecuted for:%d"%count+" times"



# *******************训练开始******************************
# 保存x方便显示
x_src = x.copy()
sgd(theta,rate,x,y,m,n)

# 显示结果
print theta
z = np.linspace(0,x_max,1000)
w = theta[0,0]+theta[1,0]*z
plt.figure()
plt.scatter(x_src[0:m,1:2],y,10,c='y')
plt.plot(z,w)
plt.show()