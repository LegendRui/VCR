# !usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2018-05-22
@author: mistery
@version: 1.0.1
'''

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

DEBUG = True

# imgs_path = './train/3000_6_character/'
# for img_name in os.listdir(imgs_path):
# 	print img_name

# img = cv2.imread('./train/3000_6_character/Z7ID97.bmp', cv2.IMREAD_GRAYSCALE)
# cv2.imshow(img_name, img)
# cv2.waitKey(0)
# cv2.destroyWindow(img_name)


# x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
# y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
# abs_x = cv2.convertScaleAbs(x)
# abs_y = cv2.convertScaleAbs(y)
# sobel_img = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
# cv2.imshow('sobel', sobel_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# gauss_img = cv2.GaussianBlur(img, (5, 5), 1.5)
# cv2.imshow('gaussian', gauss_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ret, binary_trunc_img = cv2.threshold(img , 192, 255, cv2.THRESH_TRUNC)
# cv2.imshow('binary', binary_trunc_img)
# cv2.waitKey(0)
# cv2.destroyWindow('binary')

# w, h = binary_trunc_img.shape[: 2]
# binary_img = np.zeros((w, h), np.uint8)
# for i in range(w):
# 	for j in range(h):
# 		if binary_trunc_img[i, j] < 192:
# 			binary_img[i, j] = 0
# 		else:
# 			binary_img[i, j] = 255
# cv2.imshow('binary', binary_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# w, h = sobel_img.shape[:2]

# kernel = np.ones((3, 3), np.uint8)
# open_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
# cv2.imshow('open', open_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# w, h = binary_img.shape[:2]
# bi_copy = np.zeros((w, h), np.uint8)
# for i in range(w):
# 	for j in range(h):
# 		if i == 0 or i == w - 1 or j == 0 or j == h - 1:
# 			bi_copy[i, j] = 0
# 		elif open_img[i, j] != 0:
# 			sum = 0
# 			for m in xrange(-1, 2):
# 				for n in xrange(-1, 2):
# 					if m != 0 and n != 0:
# 						sum += open_img[i + m, j + n]
# 			if sum > 255 * 2:
# 				bi_copy[i, j] = 255
# cv2.imshow('binary after procession', bi_copy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



def get_bin_img(img):
	_, bin_trunc_img = cv2.threshold(img , 192, 255, cv2.THRESH_TRUNC)
	w, h = bin_trunc_img.shape[: 2]
	bin_img = np.zeros((w, h), np.uint8)
	for i in range(w):
		for j in range(h):
			if i == 0 or j == 0 \
			or i == w - 1 or j == h - 1:
				bin_img[i, j] = 255
			elif bin_trunc_img[i, j] < 192:
				bin_img[i, j] = 0
			else:
				bin_img[i, j] = 255
	return bin_img


def img_split(img):
	h, w = img.shape[: 2]
	# print h, w
	hist = []
	for x in range(w):
		col_sum = 0
		for y in range(h):
			if img[y, x] == 0:
				col_sum += 1
			# col_sum += img[y, x]
		hist.append(col_sum)
	
	split_points = []
	for i in range(len(hist) - 1):
		if hist[i] == 0 and hist[i + 1] != 0:
			split_points.append(i)
		if hist[i] != 0 and hist[i + 1] == 0:
			split_points.append(i + 1)
	# print hist
	# print split_points
	# plt.figure()
	# plt.plot(np.array(hist))
	# plt.show()
	sub_imgs = []
	index = 0
	while index < len(split_points):
		x_begin, x_end = split_points[index] - 1, split_points[index + 1] + 1
		# print x_begin, x_end
		sub_img = img[:, x_begin : x_end]
		sub_img = cv2.resize(sub_img, (h, 18), interpolation=cv2.INTER_CUBIC)
		sub_imgs.append(sub_img)
		index += 2

	return sub_imgs

def generate_train_imgs():
	img_path = './train/3000_6_character/'
	for img_name in os.listdir(img_path):
		img = cv2.imread(img_path + img_name, cv2.IMREAD_GRAYSCALE)
		bin_img = get_bin_img(img)
		sub_imgs = img_split(bin_img)
		for sub_img, ch in zip(sub_imgs, img_name.split('.')[0]):
			# cv2.imshow('sub_img', sub_img)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			if not os.path.exists('./train/charcter_' + ch):
				if DEBUG:
					print "making dir:" + './train/charcter_' + ch
				os.mkdir('./train/charcter_' + ch)

			cv2.imwrite('./train/charcter_' + ch + '/' + img_name.split('.')[0] + '_' + ch + '.jpg', sub_img)
			print 'saving ' + img_name.split('.')[0] + '_' + ch + '.jpg to ' + './train/charcter_' + ch + '/'
		# print ''

def main():
	generate_train_imgs()
	# img_path = './train/3000_6_character/J0GL71.bmp'
	# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	# bin_img = get_bin_img(img)
	# # cv2.imshow('binary_img', bin_img)
	# # cv2.waitKey(0)
	# # cv2.destroyAllWindows()
	# sub_imgs = img_split(bin_img)
	# for sub_img in sub_imgs:
	# 	cv2.imshow('sub_img', sub_img)
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()



if __name__ == '__main__':
	main()