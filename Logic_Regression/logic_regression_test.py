# coding:utf-8
'''
    - Created with Sublime Text 2.
    - User: yoyoyohamapi
    - Date: 2015-01-09
    - Time: 14:39:03
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
b = np.loadtxt("data.txt")

# 特征集
train_set_x = np.ones([b.shape[0],b.shape[1]-1+1],dtype=float)

# m:样本数
# n:特征数
m,n = train_set_x.shape

# 初始化训练样本
train_set_x[0:m,1:2] = b[0:m,0:1]
train_set_x[0:m,2:3] = b[0:m,1:2]

# 分类向量
train_set_y = b[0:m,2:3]

# 初始化theta
theta = np.zeros([n,1],dtype=float)
theta[0] = -50

# 迭代次数
count = 0

# 定义训练参数
options = {
	'alpha':0.001,
	'max_loop':5000,
	'eplison':0.001,
	'debug':True,
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
showTrainResult(train_set_x,train_set_y,theta)
