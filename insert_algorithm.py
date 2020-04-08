# This part implements the details of insert algorithm in python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import operator
from picture_operation import * 

def img_preprocess(filename):

	'''
	input: string
	output: list
	'''
	img = cv2.imread(filename)
	img = img[:, :, 1] 
	img = img.tolist()
	return img

# Fast fourier transform, return list type
def FFT_operation(img):

	'''
	input: list
	output: list
	'''
	img_fft = np.fft.fft2(img)
	img_fft = img_fft.tolist()
	return img_fft

# Inverse fast fourier transform, return list type
def IFFT_operation(img):

	'''
	input: list
	output: list
	'''
	f1shift = np.fft.ifft2(img)
	img_ifft = f1shift.tolist()
	return img_ifft

# Insert data to the matrix after fft

def matrix_insert(watermark, img_fft):

	'''
	insert_matrix: The input n*n matrix
	img_fft: The matrix after fft
	input: list, list
	output: list
	'''

	x1 = np.array(watermark).shape[0]
	y1 = np.array(watermark).shape[1]

	x2 = np.array(img_fft).shape[0]
	y2 = np.array(img_fft).shape[1] 

	insert = 10000
	for i in range(1,x1+1):
		for j in range(1,y1+1):
			if operator.eq(watermark[i-1][j-1], [255, 255, 255]):
				img_fft[x2//2-i][y2//2-j] = insert + abs(img_fft[x2//2-i][y2//2-j].real) + img_fft[x2//2-i][y2//2-j] - img_fft[x2//2-i][y2//2-j].real 
				img_fft[x2//2+i][y2//2+j] = insert + abs(img_fft[x2//2+i][y2//2+j].real) + img_fft[x2//2+i][y2//2+j] - img_fft[x2//2+i][y2//2+j].real

	return img_fft


# Watermark reduction according to the transformed data
def watermark_back(result1, watermark_path):
	'''
	result1: The image with watermark inserted
	input: list, string
	output: list
	'''

	result1 = FFT_operation(result1)

	# Return the length and width of watermark
	length1 = cv2.imread(watermark_path).shape[0]
	width1 = cv2.imread(watermark_path).shape[1]

	# Establish the matrix to store the extracted watermark
	back_watermark = [[0] * width1 for _ in range(length1)]

	# Return the length and width of original image
	length2 = np.array(result1).shape[0]
	width2 = np.array(result1).shape[1]

	for i in range(1,length1+1):
		for j in range(1,width1+1):
			if result1[length2//2-i][width2//2-j].real > 2000:
				back_watermark[i-1][j-1] = [255, 255, 255]
			else:
				back_watermark[i-1][j-1] = [0, 0, 0]

	back_watermark = shuffle_reduction(back_watermark, 10)

	return back_watermark


# Insert watermark
def watermark_operation(watermark_back_path, img_info, watermark_info):

	'''
	input: string, list, list
	output: list
	'''

	img_fft = FFT_operation(img_info)


	# Shuffle the watermark and generate substitute matrix
	shuffled_watermark = shuffle_result(watermark_info, 10)
	cv2.imwrite(watermark_back_path, np.array(shuffled_watermark))

	# Insert watermark
	result = matrix_insert(shuffled_watermark, img_fft)

	# Original picture reduction 
	img_ifft = IFFT_operation(result)

	return img_ifft
