# -*- coding:utf-8 -*-
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 

# 实现置乱
def Arnold(list,x,y,data):
	new_data = [[0] * x for _ in range(y)]
	A = np.array(list).reshape(2,2)
	N = x
	for i in range(x):
		for j in range(y):
			m = np.array([i, j]).T
			n = np.matmul(A, m).T
			index = n.tolist()
			new_data[index[0]%N][index[1]%N] = data[i][j]
	return new_data

# 置乱处理
def shuffle_result(data, rotate_time):
	r_Matrix = [1,1,1,2]
	for i in range(rotate_time):
		data = Arnold(r_Matrix, np.array(data).shape[0], np.array(data).shape[1], data)
	return data

# 采用逆矩阵进行还原处理
def shuffle_reduction(data, rotate_time):
	r_Matrix = [2,-1,-1,1]
	for i in range(rotate_time):
		data = Arnold(r_Matrix, np.array(data).shape[0], np.array(data).shape[1], data)
	return data
