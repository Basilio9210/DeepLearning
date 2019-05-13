# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:08:16 2018

@author: Familiamadcas2
"""


import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import pickle

test_dataset = h5py.File('C:\\Users\\Familiamadcas2\\Documents\\Implementaciones\\dataset_animales\\animales_test_dataset.h5', "r")
train_dataset = h5py.File('C:\\Users\\Familiamadcas2\\Documents\\Implementaciones\\dataset_animales\\animales_train_dataset.h5', "r")
print(list(test_dataset))
print(list(train_dataset))

train_set_x_orig = np.array(train_dataset["train_set_x"]) # your train set features
test_set_x_orig = np.array(train_dataset["train_set_y"]) # your train set labels
#
train_set_y_orig = np.array(test_dataset["test_set_x"]) # your test set features
test_set_y_orig = np.array(test_dataset["test_set_y"]) # your test set labels

batch_size=640
numFilesTrain= int(len(train_set_x_orig)/batch_size + 1)
numFilesTest= int(len(test_set_x_orig)/batch_size + 1)


index=np.where(train_set_y_orig==21)

train_set_y_orig=np.delete(train_set_y_orig,index[0],axis=0)
train_set_x_orig=np.delete(train_set_x_orig,index[0],axis=0)

index=np.where(test_set_y_orig==21)

test_set_y_orig=np.delete(test_set_y_orig,index[0],axis=0)
test_set_x_orig=np.delete(test_set_x_orig,index[0],axis=0)

#
#for i in range(numFilesTrain):
#    if(i==numFilesTrain-1):
#        X_train= train_set_x_orig[i*batch_size:, ...]
#        Y_train= train_set_y_orig[i*batch_size:, ...]
#    else:
#        X_train= train_set_x_orig[i*batch_size:(i+1)*batch_size, ...]
#        Y_train= train_set_y_orig[i*batch_size:(i+1)*batch_size, ...]
#        
#    with h5py.File('animales_train_dataBatch_' + str(i) + '.h5','w') as h5data:
#        h5data.create_dataset('data',data=X_train)
#        h5data.create_dataset('labels',data=Y_train)
        
for i in range(numFilesTest):
    if(i==numFilesTest-1):
        X_test= test_set_x_orig[i*batch_size:, ...]
        Y_test= test_set_y_orig[i*batch_size:, ...]
    else:
        X_test= test_set_x_orig[i*batch_size:(i+1)*batch_size, ...]
        Y_test= test_set_y_orig[i*batch_size:(i+1)*batch_size, ...]
        
    with h5py.File('animales_test_dataBatch_' + str(i) + '.h5','w') as h5data:
        h5data.create_dataset('data',data=X_test)
        h5data.create_dataset('labels',data=Y_test)
        
train_dataset.close()
