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

import warnings
warnings.filterwarnings("ignore")


class Lenet(object):
	def __init__(self):
		self.num_of_train_image = 0
		self.num_of_class = 0
		self.train_image_path = './train_image/'
		self.test_image_path = './test/'
		self.model_path = './model'

		self.labels=[]
		sub_path = os.listdir(self.test_image_path)
		for sp in sub_path:
			self.labels.append(sp[-1])
		sub_train_image_path = os.listdir(self.train_image_path)
		self.num_of_class = len(sub_train_image_path)
		# self.read_train_image()
		# self.net()
		# self.train()

	def read_train_image(self):
		sub_train_image_path = os.listdir(self.train_image_path)
		self.num_of_class = len(sub_train_image_path)
		for path in sub_train_image_path:
			if os.listdir(self.train_image_path + path):
				files = os.listdir(self.train_image_path + path)
				self.num_of_train_image += len(files)

		self.input_images = np.array([[0] * 28 * 28 \
						for i in range(self.num_of_train_image)])
		self.input_labels = np.array([[0] * self.num_of_class\
						for i in range(self.num_of_train_image)])

		# read image
		index = 0
		for i, path in enumerate(sub_train_image_path):
			for file_name in os.listdir(self.train_image_path + path):
				img = cv2.imread(self.train_image_path + path \
				 		+ '/' + file_name, cv2.IMREAD_GRAYSCALE)
				
				height, width = img.shape[:2]
				for h in range(height):
					for w in range(width):
						if img[h, w] > 230:
							self.input_images[index][w + h * width] = 0
						else:
							self.input_images[index][w + h * width] = 1
				self.input_labels[index][i] = 1
				index += 1

	def build_net(self):

		self.x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
		self.y_ = tf.placeholder(tf.float32, shape=[None, self.num_of_class])

		self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

		self.W_conv1 = tf.Variable(tf.truncated_normal([7, 7, 1, 32], stddev=0.1))
		self.b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

		self.L1_conv = tf.nn.conv2d(self.x_image, self.W_conv1, 
								strides=[1, 1, 1, 1], padding="SAME")
		self.L1_relu = tf.nn.relu(self.L1_conv + self.b_conv1)
		self.L1_pool = tf.nn.max_pool(self.L1_relu, ksize=[1, 2, 2, 1], 
								strides=[1, 2, 2, 1], padding="SAME")

		# define variables and ops of the second convolution layer
		self.W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
		self.b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

		self.L2_conv = tf.nn.conv2d(self.L1_pool, self.W_conv2, 
								strides=[1, 1, 1, 1], padding="SAME")
		self.L2_relu = tf.nn.relu(self.L2_conv + self.b_conv2)
		self.L2_pool = tf.nn.max_pool(self.L2_relu, ksize=[1, 2, 2, 1], 
								strides=[1, 2, 2, 1], padding="SAME")

		# define full connection layer
		self.W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], 
								stddev=0.1))

		self.b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
		self.h_pool2_flat = tf.reshape(self.L2_pool, [-1, 7 * 7 * 64])
		self.h_fc1 = tf.nn.relu(tf.matmul(
							self.h_pool2_flat, self.W_fc1) + self.b_fc1)

		# dropout
		self.keep_prob = tf.placeholder(tf.float32)
		self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

		# readout layer
		self.W_fc2 = tf.Variable(tf.truncated_normal(
								[1024, self.num_of_class], stddev=0.1))
		self.b_fc2 = tf.Variable(tf.constant(0.1, shape=[self.num_of_class]))

		self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

		# define optimizer and training op
		cross_entropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(
				labels=self.y_, logits=self.y_conv))
		self.train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(
				self.y_conv, 1), tf.argmax(self.y_, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		
	def train(self):
	

		saver = tf.train.Saver()
		with tf.Session() as self.sess:
			self.sess.run(tf.global_variables_initializer())
			print "read {0} input images, {1} labels".format(
				self.num_of_train_image, self.num_of_class)

			# set number of inputs and iterations of each train
			batch_size = 60
			iterations = 100
			batches_count = int(self.num_of_train_image / batch_size)
			remainder = self.num_of_train_image % batch_size
			print ("seperate the data set to {0} parts. " + \
					"The first {1} parts include {2} datas, " + \
					"and the last one includes {3} datas.").format(
						batches_count+1, batches_count, batch_size, remainder)
			# train
			for it in range(iterations):
				# change input array to np.array
				for n in range(batches_count):
					self.train_step.run(
						feed_dict={self.x: self.input_images[n * batch_size : \
															(n + 1) * batch_size],
									self.y_: self.input_labels[n * batch_size: \
															(n + 1) * batch_size],
									self.keep_prob: 0.5})
				if remainder > 0:
					start_index = batches_count * batch_size
					self.train_step.run(
						feed_dict={self.x: self.input_images[start_index : \
														self.num_of_train_image - 1],
									self.y_: self.input_labels[start_index :\
														self. num_of_train_image - 1],
									self.keep_prob: 0.5})			

				# after 5 iterations, check whether the accuracy achieve 100%
				# if true, quit the iteration loop
				iterate_accuracy = 0
				if it % 5 == 0:
					iterate_accuracy = self.accuracy.eval(
						feed_dict={self.x: self.input_images, 
									self.y_: self.input_labels, 
									self.keep_prob: 1.0})
					print "iteration %d: accuracy %s" % (it, iterate_accuracy)

					if iterate_accuracy >= 1:
						break

			print 'Training completed!'
			saver.save(self.sess, self.model_path + '/model')

	# def restore(self):
	def predict(self, img):
		self.saver = tf.train.Saver()
		#with tf.Session() as self.sess:
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		ckpt = tf.train.get_checkpoint_state(self.model_path)
		if ckpt and ckpt.model_checkpoint_path:
				self.saver.restore( self.sess, ckpt.model_checkpoint_path)

	
		bin_img = gti.get_bin_img(img)
		sub_imgs = gti.img_split(bin_img)

		input_images = np.array([[0] * 28 * 28 for i in range(len(sub_imgs))])

		index = 0
		for sub_img in sub_imgs:
			sub_img = cv2.resize(sub_img, (28, 28), interpolation=cv2.INTER_CUBIC)
			height, width = sub_img.shape[:2]
			for h in range(height):
				for w in range(width):
					if sub_img[h, w] > 230:
						input_images[index][w + h * width] = 0
					else:
						input_images[index][w + h * width] = 1
			index += 1
		y = self.sess.run(self.y_conv,
					feed_dict={self.x: input_images,
								self.keep_prob: 0.5})

		# self.saver.save(self.sess, self.model_path + '/model')

		ret = []
		ret.append(self.labels[np.argmax(y[0])])
		ret.append(self.labels[np.argmax(y[1])])
		ret.append(self.labels[np.argmax(y[2])])
		ret.append(self.labels[np.argmax(y[3])])
		return ret


if __name__ == '__main__':
	cnn = Lenet()
	cnn.read_train_image()
	cnn.build_net()
	cnn.train()
	# cnn.restore()
	img = cv2.imread('./test/00CB.bmp', cv2.IMREAD_GRAYSCALE)
	r = cnn.predict(img)
	print r