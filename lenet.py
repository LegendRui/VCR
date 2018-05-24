# !usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2018-05-24
@author: mistery
@version: 1.0.0
'''


from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
	@staticmethod
	def build(width, height, depth,classes):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)
		# if we are using "channels last", update the input shape
		if K.image_data_format() == "channels_first": # for tensorflow
			inputShape = (depth, height,width)
		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(20, (5, 5), padding="same", innput_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(50, (5, 5), padding='same'))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# first (and only)set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		# softmax classfier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model


import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import sys
sys.path.append('..')
# from net.lenet import LeNet