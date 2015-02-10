#coding:utf-8

'''
    - Created with Sublime Text 2.
    - User: yoyoyohamapi
    - Date: 2015-01-27
    - Time: 14:34:01
    - Contact: yoyoyohamapi.com
'''

import numpy as np
import random
import os
# 定义sigmoid函数
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))



#############################
#------BP神经网络建模-------#
#############################
class BPNN:
	"""
	构造函数:
		:param x:输入特征向量
		:param hidden_num:隐层数
		:param y:分类向量
		
	"""
	def __init__(self,x,y,hidden_num):
		self.x = x
		self.y = y
		# 训练样本数
		self.m = x.shape[0]
		self.hidden_num = hidden_num
		# 各层需要添加偏置向量
		self.weights = []
		# 新增特征作为偏置
		self.x = np.column_stack( (x,np.ones([self.m,1],dtype=float)) )
		# 隐藏层神经元数目默认比特征数多1,且添加偏置
		self.hidden_node_num = self.x.shape[1] + 1 + 1
		# BP网络层数
		self.K = self.hidden_num + 2
		# 各层神经元数目
		self.layer_num = [self.hidden_node_num for i in range(self.K)]
		self.layer_num[0] = self.x.shape[1]
		self.layer_num[self.K-1] = self.y.shape[0]
		self.layer_num = tuple(self.layer_num)

		# 初始化权值矩阵
		self.initWeights()
		
	"""
	训练函数:
		:param options:训练参数
				1. rate:学习率
				2. alpha:权值修正系数
				3. max_loop:最大迭代次数
				4. eplison:收敛精度
	"""
	def	train(self,options=None):
		# 初始化训练选项
		rate = options['rate']
		max_loop = options['max_loop']
		eplison = options['eplison']

		##########################
		#####     训练开始   #####
		##########################
		# for i in range(self.m):
		# 	print 'i is %d'%i
		# 	# 各层输出
		# 	a = []

		# 	# 各层误差
		# 	d = [1000 for num in range(self.K)]
		# 	# Forward Propagation 计算各层输出
		# 	for k in range(self.K):
		# 		if k==0 :
		# 			a.append(self.x[i].T)
		# 		else:
		# 			z = np.dot(self.weights[k-1],a[k-1])
		# 			a.append( sigmoid(z) )
		# 	# Back Propagation 修正误差
		# 	for k in range(self.K-1,0,-1):
		# 		if k==self.K-1:
		# 			d[k] = np.multiply(np.multiply(a[k],(1.0-a[k])),(a[k]-self.y[:,i]))
		# 		else:
		# 			d[k] = np.multiply(np.multiply(a[k],(1.0-a[k])),np.dot(self.weights[k].T,d[k+1]))
		# 		# 修正权值
		# 		for t in range(max_loop):
		# 			# 暂存上一次权值矩阵
		# 			last = self.weights[k-1].copy()
		# 			self.weights[k-1] = self.weights[k-1] - rate*np.dot(d[k],a[k-1].T)
		# 			if abs(self.weights[k-1]-last).max() < eplison:
		# 				break
		# 		print t
		for t in range(max_loop):
			delta = [0 for num in range(self.K-1)]
			for i in range(self.m):
				# 各层输出
				a = []

				# 各层误差
				d = [1000 for num in range(self.K)]
				# Forward Propagation 计算各层输出
				for k in range(self.K):
					if k==0 :
						a.append(self.x[i].T)
					else:
						z = np.dot(self.weights[k-1],a[k-1])
						a.append( sigmoid(z) )
				# Back Propagation 修正误差
				for k in range(self.K-1,0,-1):
					if k==self.K-1:
						d[k] = np.multiply(np.multiply(a[k],(1.0-a[k])),(a[k]-self.y[:,i]))
					else:
						d[k] = np.multiply(np.multiply(a[k],(1.0-a[k])),np.dot(self.weights[k].T,d[k+1]))
					# 修正权值
					delta[k-1] = delta[k-1] + np.dot(d[k],a[k-1].T)
			eplisons = [1000 for k in range(self.K-1)]
			for k in range(self.K-1):
				last = self.weights[k].copy()
				self.weights[k] = last - rate*((1.0/self.m)*delta[k])
				eplisons[k] = abs(self.weights[k]-last).max()
			if max(eplisons) < eplison:
				break
		print t

	"""  
	初始化权值矩阵，各权值落在0-1内
	"""
	def initWeights(self):
		for i in range(self.K - 1):
			m = self.layer_num[i+1]
			n = self.layer_num[i]
			weights = np.matrix(np.random.rand(m,n))
			# 权值阵最后一行为阀值theta
			weights[:,n-1] = weights[:,n-1]*0 + 1
			self.weights.append( weights)

if __name__ == '__main__':  
	train_b = np.loadtxt("train.txt")
	train_x = train_b[:,0:4]
	train_y = train_b[:,4:7]
	train_x = np.matrix(train_x)
	train_y = np.matrix(train_y)
	train_y = train_y.T

	bpnn = BPNN(train_x,train_y,2)

	# 定义训练参数
	options = {
		'rate':0.01,
		'max_loop':10000,
		'eplison':0.00001,
	}

	bpnn.train(options)
	test_b = np.loadtxt("test.txt")
	test_x = test_b[:,0:4]
	test_x = np.matrix(test_x)
	test_x = np.column_stack( (test_x,np.ones([test_x.shape[0],1],dtype=float)) )
	z1 = np.dot(bpnn.weights[0],bpnn.x[30].T)
	a1 = sigmoid(z1)
	z2 = np.dot(bpnn.weights[1],a1)
	a2 = sigmoid(z2)
	z3 = np.dot(bpnn.weights[2],a2)
	pred_y = sigmoid(z3)
	print pred_y

	z1 = np.dot(bpnn.weights[0],test_x[60].T)
	a1 = sigmoid(z1)
	z2 = np.dot(bpnn.weights[1],a1)
	a2 = sigmoid(z2)
	z3 = np.dot(bpnn.weights[2],a2)
	pred_y1 = sigmoid(z3)
	print pred_y1
