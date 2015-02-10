# coding:utf-8
'''
    - Created with Sublime Text 2.
    - User: yoyoyohamapi
    - Date: 2015-01-13
    - Time: 09:15:22
    - Contact: yoyoyohamapi.com
'''

###################################
#                                 #
#      测试Logic Regression       #
#                                 #
###################################

import numpy as np
import time
from logic_regression import *

# 载入数据
b = np.loadtxt("data2.txt")

# 特征集,28维
train_set_x = np.ones([b.shape[0],28],dtype=float)

# m:样本数
# n:特征数
m,n = train_set_x.shape

# 初始化训练样本,28维
train_set_x[0:m,1:2] = b[0:m,0:1] #x1
train_set_x[0:m,2:3] = b[0:m,1:2] #x2
x1 = train_set_x[0:m,1:2].copy()
x2 = train_set_x[0:m,2:3].copy()

degree = 6
index = 0
for i in range(1,degree+1):
	for j in range(i+1):
		index = index + 1
		train_set_x[0:m,index:index+1] = pow(x1,i-j)*pow(x2,j)



# 分类向量
train_set_y = b[0:m,2:3]

# 初始化theta
theta = np.zeros([n,1],dtype=float)

# 迭代次数
count = 0

# 定义训练参数
options = {
	'alpha':0.9,
	'max_loop':5000,
	'eplison':0.001,
	'debug':True,
	'theLambda':1,
	'method':'BGD'
}

#########################
#--------训练开始-------#
#########################

print 'Trainning start...................'
# 计时
start = time.clock()
# 训练 
theta,count = trainLogicRegression(train_set_x,train_set_y,theta,options)
# 停止计时
end = time.clock()
time_cost = end - start

#########################
#--------显示结果-------#
#########################
print 'Trainning end,the number of iteration is:%d'%count
print 'Time consumed:%ld'%time_cost+'s'
print 'Vector theta is:'
print theta
showTrainResult(train_set_x,train_set_y,theta,options['theLambda'])
