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
		self.b = []
		# 隐藏层神经元数目默认比特征数多1,且添加偏置
		self.hidden_node_num = self.x.shape[1] + 1		
		# BP网络层数
		self.K = self.hidden_num + 2
		# 各层神经元数目
		self.layer_num = [self.hidden_node_num for i in range(self.K)]
		self.layer_num[0] = self.x.shape[1]
		self.layer_num[self.K-1] = self.y.shape[0]
		self.layer_num = tuple(self.layer_num)

		# 初始化权值矩阵,以及偏置向量
		self.initWeightsB()
		
	"""
	训练函数:
		:param options:训练参数
				1. rate:学习率
				2. alpha:权值修正系数
				3. max_loop:最大迭代次数
				4. eplison:收敛精度
				5. theLambda:for regularazation
	"""
	def	train(self,options=None):
		# 初始化训练选项
		rate = options['rate']
		max_loop = options['max_loop']
		eplison = options['eplison']
		theLambda = options['theLambda']
		for t in range(max_loop):
			# delta_W计算权值矩阵梯度偏导数所用的增量
			delta_W = [0 for num in range(self.K-1)]
			# delta_b计算偏置向量梯度偏导数所用的增量
			delta_b = [0 for num in range(self.K-1)]
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
						z = np.dot(self.weights[k-1],a[k-1])+self.b[k-1]
						a.append( sigmoid(z) )
				# Back Propagation 修正误差
				for k in range(self.K-1,0,-1):
					if k==self.K-1:
						d[k] = np.multiply(np.multiply(a[k],(1.0-a[k])),(a[k]-self.y[:,i]))
					else:
						d[k] = np.multiply(np.multiply(a[k],(1.0-a[k])),np.dot(self.weights[k].T,d[k+1]))
					# 修正权值
					delta_W[k-1] = delta_W[k-1] + np.dot(d[k],a[k-1].T)
					delta_b[k-1] = delta_b[k-1] + d[k]
			
			for k in range(self.K-1):
				self.weights[k] = self.weights[k] - rate*((1.0/self.m)*delta_W[k]+theLambda*self.weights[k])
				self.b[k] = self.b[k] - rate*(1.0/self.m*delta_b[k])
			sys_error = self.J(theLambda)
			if abs(sys_error) < eplison:
				break
		print 'The iteration has done %d times'%t

	"""  
	初始化权值矩阵，各权值落在随机分布在0附近
	"""
	def initWeightsB(self):
		for i in range(self.K - 1):
			m = self.layer_num[i+1]
			n = self.layer_num[i]
			weights = np.matrix(np.random.normal(scale=0.01,size=[m,n]))
			# 权值阵最后一行为阀值theta
			self.weights.append( weights)
			b = (np.random.normal(scale=0.01,size=[m,1]))
			self.b.append(b)
	
	"预测函数"
	def h(self,x):
		a = []
		for k in range(self.K):
			if k==0 :
				a.append(x)
			else:
				z = np.dot(self.weights[k-1],a[k-1])+self.b[k-1]
				a.append( sigmoid(z) )
		return a[self.K-1]

	"""
	整体代价函数
	"""
	def J(self,theLambda):
		result =0.0
		for i in range(self.m):
			error = self.h(self.x[i].T) - self.y[:,i]
			result = result + 1.0/2.0*(np.multiply(error,error)).sum()
		w_sum = 0.0
		for i in range(len(self.weights)):
			w_sum = w_sum + np.sum(np.multiply(self.weights[i],self.weights[i]))
		return 1.0/self.m*result-theLambda/(2.0*self.m)*w_sum


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
		'rate':5,
		'max_loop':2000,
		'eplison':0.02,
		'theLambda':0.0
	}
	###############################
	########    训练开始        ####
	###############################
	bpnn.train(options)
	test_b = np.loadtxt("test.txt")
	test_x = test_b[:,0:4]
	test_y = test_b[:,4:7]
	test_x = np.matrix(test_x)
	test_y = np.matrix(test_y)
	test_y = test_y.T


	print 'the error of test 1 is:'
	print abs(bpnn.h(test_x[10].T) - test_y[:,10])
	print 'the error of test 2 is:'
	print abs(bpnn.h(test_x[30].T) - test_y[:,30])
	print 'the error of test 3 is:'
	print abs(bpnn.h(test_x[40].T) - test_y[:,40])
	print 'the error of test 4 is:'
	print abs(bpnn.h(test_x[60].T) - test_y[:,60])
	print 'the error of test 5 is:'
	print abs(bpnn.h(test_x[70].T) - test_y[:,70])
