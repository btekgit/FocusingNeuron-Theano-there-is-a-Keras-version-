#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 12:43:12 2018

@author: btek
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import RMSprop, SGD
from data_utils import load_dataset_mnist
batch_size = 512
num_classes = 10
epochs = 5


# the data, split between train and test sets
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_mnist(folder='../datasets/mnist/')


#(x_train, y_train), (x_test, y_test) = load_dataset_mnist()

x_train = x_train.reshape(50000, 784)
x_val = x_val.reshape(10000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(800, activation='linear', input_shape=(784,),name='dense1'))
model.add(BatchNormalization(name='bn1'))
model.add(Activation('relu'))
model.add(Dropout(0.25, name='drp1'))
model.add(Dense(800, activation='linear', name='dense2'))
model.add(BatchNormalization(name='bn2'))
model.add(Activation('relu'))
model.add(Dropout(0.25,name='drp2'))
model.add(Dense(num_classes, activation='softmax',name='out'))
model.get_config()
model.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

#for layer in model.layers:

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# now loading our networks
import numpy as np
import os


import numpy as np
import matplotlib.pyplot as plt
#os.environ['THEANO_FLAGS']='device=cpu, gpuarray.preallocate=.1'
os.environ['MKL_THREADING_LAYER']='GNU'
import time
from focusing import U_numeric

#u = np.load("outputs/ESNN/mnist9/focused_s/mnist_model_focused_mlp:2,800,.25,.50_20180720-104103.npz")
u2 = np.load('outputs/ESNN/mnist10mnist_model_focused_mlp:2,800,0.25,0.25_20180808-185004.npz')
print(u2['arr_0'])
mu_1 = u2['arr_1'][0]
si_1 = u2['arr_1'][1]
weights_1= u2['arr_1'][2]
bias_1 = u2['arr_1'][3]
bn_1_beta = u2['arr_1'][4]
bn_1_gamma = u2['arr_1'][5]
bn_1_mean = u2['arr_3'] [1]
bn_1_std = u2['arr_3'] [2]


mu_2 = u2['arr_1'][6]
si_2 = u2['arr_1'][7]
weights_2= u2['arr_1'][8]
bias_2 = u2['arr_1'][9]
bn_2_beta = u2['arr_1'][10]
bn_2_gamma = u2['arr_1'][11]
bn_2_mean = u2['arr_3'] [4]
bn_2_std = u2['arr_3'] [5]


weights_out = u2['arr_1'][12]
bias_out = u2['arr_1'][13]


def prune(x):
    return np.maximum(0,x-1e-6)
    
fis_1 = U_numeric(np.linspace(0,1,weights_1.shape[0]),mu_1,si_1,1)
fis_1 = prune(fis_1)
fis_2 = U_numeric(np.linspace(0,1,weights_2.shape[0]),mu_2,si_2,1)
fis_2 = prune(fis_2)
fw1 = (fis_1.T*weights_1)
fw2 = (fis_2.T*weights_2)

#fw1[fw1<1e-14]=0
#fw2[fw2<1e-14]=0

dense1= model.get_layer('dense1')
dense1.set_weights([fw1, bias_1])
#dense1.trainable=False

bn1=model.get_layer('bn1')
bn1_params= bn1.get_weights()
bn1.set_weights([bn_1_gamma, bn_1_beta, bn_1_mean, bn_1_std])
#bn1.trainable=False

dense2= model.get_layer('dense2')
dense2.set_weights([fw2, bias_2])
#dense2.trainable=False

bn2=model.get_layer('bn2')
bn2_params= bn2.get_weights()
bn2.set_weights([bn_2_gamma, bn_2_beta, bn_2_mean, bn_2_std])
#bn2.trainable=False

out= model.get_layer('out')
out.set_weights([weights_out,bias_out])
#out.trainable=False

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

epochs = 1

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
