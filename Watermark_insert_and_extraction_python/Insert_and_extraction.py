# This part implements the details of the complete procedure of generating training set
'''
Module name list:
opencv-python
numpy
pillow
Attention: All paths here are supposed to be replaced by your own paths
'''
import cv2
from picture_operation import *
from insert_algorithm import *
from png_to_jpg import *
import ast


def preprocess_on_insert(result):
	length = np.array(result).shape[0]
	width = np.array(result).shape[1]
	for i in range(length):
		for j in range(width):
			result[i][j] = result[i][j].real
	return result

def get_watermark():
	# Get the watermark from stationary path
	# REPLACE IT!
	watermark_path = your_watermark_path
	watermark_info = cv2.imread(watermark_path)
	watermark_info = watermark_info.tolist()
	return watermark_info

def preprocess_img(filename):
	img_info = cv2.imread(filename)
	img_info2 = img_info[:,:,1]
	img_info2 = img_info2.tolist()
	return img_info2

def output_img(filename, result1):
	img_info = cv2.imread(filename)
	img_info[:,:,1] = result1
	# Appoint the output path
	# REPLACE IT!
	cv2.imwrite(your_output_path, np.array(img_info))



def deipel_img(filename):
	img_info = cv2.imread(filename)
	img = img_info[:,:,1]
	length = np.array(img).shape[0]
	width = np.array(img).shape[1]
	for i in range(length):
		for j in range(width):
			if img[i][j] >= 240 and j < width - 1:
				img[i][j] = img[i][j+1]
	return img

def revise_watermark(result):
	# REPLACE IT!
	revise_path = your_own_revise_path
	with open(revise_path,"r") as f:
		info = f.read()
	list_list = ast.literal_eval(info)
	height = np.array(result).shape[0]
	width = np.array(result).shape[1]
	for i in range(height):
		for j in range(width):
			for n in range(len(list_list)):
				list =[]
				list.append(i)
				list.append(j)
				if operator.eq(list, list_list[n]):
					result[i][j] = [255,255,255]
	return result


def main():
	while(1):
		choice = int(input("Please input the operation option: \n1.Insert watermark\n2.Extract watermark from image\n3.Exit\n"))
		if choice == 1:
			filepath = path_of_image_after_insert
			watermark_info = get_watermark()
			filename = input("Please input the path of image needing watermark inserting:\n")
			img_info2 = preprocess_img(filename)
			# Insert watermark
			result1 = watermark_operation(img_info2, watermark_info)
			result1 = preprocess_on_insert(result1)
			output_img(filename, result1)
			print("Insert finished!")
			print("Revise start!")
			img = deipel_img(filepath)
			output_img(filepath, img)
			print("Finished!")
		elif choice == 2:
			# REPLACE IT!
			watermark_path = your_own_watermark_path
			# Watermark after extraction
			# One folder is needed to store the extracted watermark, we suppose it as watermark_back
			# REPLACE IT!
			watermark_back_path = you_own_watermark_back_path # Include the watermark_back folder
			inserted_img = input("Please input the extraction path:")
			img_after_insert = preprocess_img(inserted_img)
			result = watermark_back(img_after_insert, watermark_path)
			revise_watermark(result)
			cv2.imwrite(watermark_back_path, np.array(result))
			print("Extraction finished!")
			flag = input("Transform the png format to jpg format? yes or no\n")
			if flag == "yes" or "y" or "Y":
				# Transform all 100 png format pictures in watermark_back folder to jpg format
				for i in range(100):
					# REPLACE IT!
					path = your_own_watermark_back_folder + str(i) + ".png"
					transimg(path)
		else:
			break

if __name__ == '__main__':
	main()

