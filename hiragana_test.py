# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 20:09:27 2018

@author: chandler
"""

import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split
import cv2 as cv
import os


nb_classes = 72
# input image dimensions
img_rows, img_cols = 32, 32
#img_rows, img_cols = 127, 128
#img_rows, img_cols = 11, 10



ary = np.load("hiragana.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32) / 15
X_train = np.zeros([nb_classes * 160, img_rows, img_cols], dtype=np.float32)
for i in range(nb_classes * 160):
#    tmp = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')
#    tmp = cv.resize(ary[i],(img_cols,img_rows),cv.INTER_CUBIC)
#    _,X_train[i]= cv.threshold(tmp,0.05,1,cv.THRESH_BINARY)
#    X_train[i]=tmp
    _,tmp = cv.threshold(ary[i],0.05,1,cv.THRESH_BINARY)
    X_train[i] = cv.resize(ary[i],(img_cols,img_rows),cv.INTER_CUBIC)
#    _,X_train[i] = cv.threshold(tmp,0.05,1,cv.THRESH_BINARY)
#    ===================
    # X_train[i] = ary[i]
#    =====================
#    X_train[i] = scipy.misc.imresize(ary[i], (img_rows, img_cols), mode='F')
Y_train = np.repeat(np.arange(nb_classes), 160)
##tmp = cv.blur(X_train[0],(3,3))
#_,tmp = cv.threshold(X_train[0],0.05,1,cv.THRESH_BINARY)
#cv.imshow('test',tmp)

for i in range(72):
    cv.imwrite(os.path.join('glance',str(i)+'.jpg'),X_train[160*i,:,:]*255)
    cv.imshow("test",X_train[160*i,:,:])

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)


np.savez('hiragana_dataset.npz',x_train=X_train,x_test=X_test,y_train=Y_train,y_test=Y_test)



#import cv2
#import numpy as np
#import os
#hiragana = np.load('hiragana_dataset.npz')
#x_train = hiragana['x_train']
#for i in range(72):
#    cv2.imwrite(os.path.join('glance',str(i)+'.jpg'),x_train[128*i,:,:])