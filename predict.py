# !usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2018-05-25
@author: mistery
@version: 1.0.0

reference:
[1]https://www.cnblogs.com/lijunjiang2015/p/7812996.html
[2]https://blog.csdn.net/huachao1001/article/details/78501928
'''

import generate_train_imgs as gti
import cv2

test_dir = './test/0A20.bmp'
img = cv2.imread(test_dir, cv2.IMREAD_GRAYSCALE)
cv2.imshow('0A20.bmp', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

bin_img = gti.get_bin_img(img)
sub_imgs = gti.img_split(bin_img)
for sub_img in sub_imgs:
    cv2.imshow('sub_img', sub_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

