# !usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2018-05-25
@author: mistery
@version: 1.0.0

reference:
[1]https://www.cnblogs.com/lijunjiang2015/p/7812996.html
[2]https://blog.csdn.net/huachao1001/article/details/78501928
[3]http://www.jb51.net/article/134623.htm
[4]https://github.com/gzdaijie/tensorflow-tutorial-samples/tree/master/mnist/v3
[5]https://juejin.im/post/5adc945e518825673027bbfb
'''


import generate_train_imgs as gti
import tensorflow as tf
import numpy as np
import cv2
import os



def predict(img):

	model_path = './model'
	num_of_class = 36
	path = './train_image/'
	labels=[]
	sub_path = os.listdir(path)
	for sp in sub_path:
		labels.append(sp[-1])

	# img = cv2.imread(test_dir, cv2.IMREAD_GRAYSCALE)
	# cv2.imshow('0A20.bmp', img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

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
	W_fc2 = tf.Variable(tf.truncated_normal([1024, num_of_class], stddev=0.1))
	b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_of_class]))

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	# define optimizer and training op
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		ckpt = tf.train.get_checkpoint_state(model_path)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		# for image in sub_imgs:
		# 	flatten_image = np.reshape(image, 28*28)
		# 	x_img = np.array([1 - flatten_image])
		# 	print x_img
		# for i in range(len(sub_imgs)):
		y = sess.run(y_conv,
					feed_dict={x: input_images,
								keep_prob: 0.5})
		ret = []
		ret.append(labels[np.argmax(y[0])])
		ret.append(labels[np.argmax(y[1])])
		ret.append(labels[np.argmax(y[2])])
		ret.append(labels[np.argmax(y[3])])
		saver.save(sess, './model/model')
	return ret
		

def main():

	for test_img_dir in os.listdir('./test/'):
		print './test/' + test_img_dir
		test_img = cv2.imread('./test/' + test_img_dir, 
							cv2.IMREAD_GRAYSCALE)
		result = predict(test_img)

		print 'The prediction result of {0} is {1} {2} {3} {4}'.format(
			test_img_dir, result[0], result[1], result[2], result[3]) 


if __name__ == '__main__':
	main()