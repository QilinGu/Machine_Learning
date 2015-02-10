#coding:utf-8
'''
    - Created with Sublime Text 2.
    - User: yoyoyohamapi吴晓军
    - Date: 2015-01-09
    - Time: 10:54:04
    - Contact: yoyoyohamapi.com
'''

###################################
#      实现Logic Regression       #
#      科学计算通过numpy完成      #
#      图像绘制通过matplotlib完成 #
###################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from sympy import *
"""
定义sigmoid函数
"""
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

"""
定义代价函数
	:param x:输入矩阵
	:param y:分类向量
	:param theta:拟合参数theta
	:return:预测精度
"""
def cost(x,y,theta):
	z = np.dot(x,theta)
	return -y*np.log(sigmoid(z)) 
	- (1-y)*np.log(1 - sigmoid(z))

""" 
定义代价函数评估
	:param x:输入矩阵
	:param y:分类向量
	:param theta:拟合参数theta
	:return:预测精度
"""
def J(x,y,theta,theLambda):
	return np.mean( cost(x,y,theta) )+theLambda*np.mean(theta)/2

"""
定义逻辑回归训练函数
	:param train_set_x:训练集--特征
	:param train_set_y:训练集--分类
	:param theta:初始化参数theta
	:param options:训练选项:
				   1.alpha 学习率
				   2.max_loop 最大迭代次数
				   3.eplison 收敛精度
				   4.debug 是否观察预测精度J的变化
				   5.method 训练方法:BGD？SGD？
	:return :拟合参数
"""

def trainLogicRegression(train_set_x,train_set_y,theta,options):
	# m:样本数
	# n:特征数
	m,n = train_set_x.shape
	
	# 迭代次数
	count = 0

	# 训练参数初始化
	alpha = options['alpha']
	max_loop = options['max_loop']
	eplison = options['eplison']
	debug = options['debug']
	method = options['method']
	theLambda = options['theLambda']

	methods = {
		'BGDdebug':BGDdebug,
		'SGDdebug':SGDdebug,
		'BGD':BGD,
		'SGD':SGD,
	}

	# 保存上次参数theta
	last = theta.copy()
	if debug:
		theta,count= methods.get(method+'debug')(train_set_x,train_set_y,theta,alpha,eplison,max_loop,count,last,m,theLambda)
	else:
		theta,count= methods.get(method)(train_set_x,train_set_y,theta,alpha,eplison,max_loop,count,last,m,theLambda)
	print "The number of iteration is :%d"%count
	return theta,count

"""
定义Batch Gradient Descent
"""
def BGDdebug(train_set_x,train_set_y,theta,alpha,eplison,max_loop,count,last,m,theLambda):
	plt.figure()
	while count <= max_loop:
		count=count+1
		diff = sigmoid(np.dot(train_set_x,theta)) - train_set_y
		theta = theta - alpha*(np.dot(train_set_x.T,diff)/m+theLambda*theta/m)
		# 绘制预测精度状况
		plt.scatter(
			count,
			J(train_set_x,train_set_y,theta,theLambda),
			marker='o',
			color='y',
			s=50)

		if np.max(abs(theta-last)) < eplison:
			break
		else:
			last = theta.copy()
	plt.show()
	return theta,count

def BGD(train_set_x,train_set_y,theta,alpha,eplison,max_loop,count,last,m,theLambda):
	while count <= max_loop:
		count=count+1
		diff = sigmoid(np.dot(train_set_x,theta)) - train_set_y
		theta = theta - alpha*(np.dot(train_set_x.T,diff)/m+theLambda*theta/m)
		if np.max(abs(theta-last)) < eplison:
			break
		else:
			last = theta.copy()
	return theta,count

"""
定义Stotochastic Gradient Descent
"""
def SGDdebug(train_set_x,train_set_y,theta,alpha,eplison,max_loop,count,last,m,theLambda):
	plt.figure()
	while count <= max_loop:
		count=count+1
		for i in range(m):
			diff = sigmoid(np.dot(train_set_x[i],theta)) - train_set_y[i]
			theta = (theta.T - alpha*(train_set_x[i]-theLambda*theta.T)*diff).T
		# 绘制预测精度状况
		plt.scatter(
			count,
			J(train_set_x,train_set_y,theta,theLambda),
			marker='o',
			color='y',
			s=50)

		if np.max(abs(theta-last)) < eplison:
			break
		else:
			last = theta.copy()
	plt.xlabel('Iteration Step')
	plt.ylabel('Cost Evaluation')
	plt.show()
	return theta,count

def SGD(train_set_x,train_set_y,theta,alpha,eplison,max_loop,count,last,m,theLambda):
	while count <= max_loop:
		count=count+1
		for i in range(m):
			diff = sigmoid(np.dot(train_set_x[i],theta)) - train_set_y[i]
			theta = (theta.T - alpha*(train_set_x[i]-theLambda*theta.T)*diff).T
		if np.max(abs(theta-last)) < eplison:
			break
		else:
			last = theta.copy()
	return theta,count

"""
定义结果绘制函数
	:param train_set_x:训练集--特征
	:param train_set_y:训练集--分类
	:param theta:参数向量
"""
def showTrainResult(train_set_x,train_set_y,theta,theLambda):
	# m:样本数
	# n:特征数
	m,n = train_set_x.shape

	x1 = np.linspace(np.min(train_set_x[0:m,1:2]),np.max(train_set_x[0:m,1:2]),256)
	x2 = np.linspace(np.min(train_set_x[0:m,2:3]),np.max(train_set_x[0:m,2:3]),256)
	x1,x2 = np.meshgrid(x1,x2)

	plt.figure()
	# 绘制样本点
	for i in range(m):
		if train_set_y[i]==0:
			plt.scatter(train_set_x[i,1],train_set_x[i,2],marker='o',color='y',s=50)
		else:
			plt.scatter(train_set_x[i,1],train_set_x[i,2],marker='*',color='r',s=50)
	# 绘制决策边界
	cs = plt.contour(x1, x2, f(x1,x2,theta), 1, colors='green', linewidth=.5)
	# 添加图示
	plt.scatter(train_set_x[0,1],train_set_x[0,2],marker='o',color='y',s=50,label="y=0")
	plt.scatter(train_set_x[0,1],train_set_x[0,2],marker='*',color='r',s=50,label="y=1")
	cs.collections[0].set_label("Decision Boundry")
	plt.title('lambda=%d'%theLambda)
	plt.legend(loc='upper left')
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()		


"""
定义决策边界多项式
"""
def f(x1,x2,theta):
	z = 0
	degree = 6
	index = 1
	for i in range(1,degree+1):
		for j in range(i+1):
			z = z+theta[index]*pow(x1,i-j)*pow(x2,j)
			index = index+1
	return z+theta[0]

 