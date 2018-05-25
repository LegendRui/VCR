# !usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2018-05-24
@author: mistery
@version: 1.0.0

reference: 
[1]http://www.cnblogs.com/jyxbk/p/7773304.html
[2]https://zhuanlan.zhihu.com/p/33739734
'''
import os
import numpy as np
import tensorflow as tf
import cv2

DEBUG = True


# count the number of images
num_of_train_image = 0
num_of_class = 0
path = './train_image/'
sub_path = os.listdir(path)
num_of_class = len(sub_path)
for p in sub_path:
	if os.path.isdir(path + p):
		files = os.listdir(path + p)
		num_of_train_image += len(files)


if DEBUG:
	print "number of train_image: ", num_of_train_image
	print "number of class: ", num_of_class


# define the dimension of train_image and array
input_images = np.array([[0] * 28 * 28 for i in range(num_of_train_image)])
input_labels = np.array([[0] * num_of_class for i in range(num_of_train_image)])

index = 0
for i, p in enumerate(sub_path):
	for file_name in os.listdir(path + p):
		img = cv2.imread(path + p + '/' + file_name, cv2.IMREAD_GRAYSCALE)
		height, width = img.shape[:2] 

		for h in range(height):
			for w in range(width):
				if img[h, w] > 230:
					input_images[index][w + h * width] = 0
				else:
					input_images[index][w + h * width] = 1
		input_labels[index][i] = 1
		index += 1

# define input node
x = tf.placeholder(tf.float32, shape=[None, 28*28])
y_ = tf.placeholder(tf.float32, shape=[None, num_of_class])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# define variables and ops of the first convonlution layer
W_conv1 = tf.Variable(tf.truncated_normal([7, 7, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

L1_conv = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding="SAME")
L1_relu = tf.nn.relu(L1_conv + b_conv1)
L1_pool = tf.nn.max_pool(L1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# define variables and ops of the second convolution layer
W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

L2_conv = tf.nn.conv2d(L1_pool, W_conv2, strides=[1, 1, 1, 1], padding="SAME")
L2_relu = tf.nn.relu(L2_conv + b_conv2)
L2_pool = tf.nn.max_pool(L2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# define full connection layer
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))

b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_pool2_flat = tf.reshape(L2_pool, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# define optimizer and training op
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print "read {0} input images, {1} labels".format(num_of_train_image, num_of_train_image)

	# set number of inputs and iterations of each train
	batch_size = 60
	iterations = 100
	batches_count = int(num_of_train_image / batch_size)
	remainder = num_of_train_image % batch_size
	print "seperate the data set to {0} parts. The first {1} parts include {2} \
			datas, and the last one includes {3} datas.".format(batches_count+1, \
											batches_count, batch_size, remainder)
	# train
	for it in range(iterations):
		# change input array to np.array
		for n in range(batches_count):
			train_step.run(
				feed_dict={x: input_images[n * batch_size : (n + 1) * batch_size],
							y_: input_labels[n * batch_size: (n + 1) * batch_size],
							keep_prob: 0.5})
		if remainder > 0:
			start_index = batches_count * batch_size
			train_step.run(
				feed_dict={x: input_images[start_index : num_of_train_image - 1],
							y_: input_labels[start_index : num_of_train_image - 1],
							keep_prob: 0.5})			

		# after 5 iterations, check whether the accuracy achieve 100%
		# if true, quit the iteration loop
		iterate_accuracy = 0
		if it % 5 == 0:
			iterate_accuracy = accuracy.eval(
				feed_dict={x: input_images, y_: input_labels, keep_prob: 1.0})
			print "iteration %d: accuracy %s" % (it, iterate_accuracy)

			if iterate_accuracy >= 1:
				break

	print 'Training completed!'