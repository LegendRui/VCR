# !usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2018-05-25
@author: mistery
@version: 1.1.0

reference:
[1]https://www.cnblogs.com/lijunjiang2015/p/7812996.html
[2]https://blog.csdn.net/huachao1001/article/details/78501928
[3]http://www.jb51.net/article/134623.htm
[4]https://github.com/gzdaijie/tensorflow-tutorial-samples/tree/master/mnist/v3
'''


import generate_train_imgs as gti
import tensorflow as tf
import numpy as np
import cv2
import os

class Lenet(object):
	def __init__(self):
		self.num_of_train_image = 0
		self.num_of_class = 0
		self.train_image_path = './train_image/'
		self.test_image_path = './test/'

	def read_train_image(self):
		sub_train_image_path = os.listdir(self.train_image_path)