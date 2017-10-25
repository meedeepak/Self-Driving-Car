# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 15:23:47 2017

@author: Deepak
"""

#import tensorflow as tf
#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()

import numpy as np
from random import shuffle
from model_structure import create_model

classifier = create_model()

classifier.load_weights('/out/model_opt.model')

train_data=np.load('/data1/final_data.npy')

for i in range(52):
    print(i)
    file_name = '/data2/training_data-{}.npy'.format(i)
    train_data_2=np.load(file_name)
    train_data=np.concatenate((train_data,train_data_2))
    
shuffle(train_data)
x=train_data[:,0]
y=train_data[:,1]
train_data=[]

x=x.tolist()
x=np.array(x)

y=y.tolist()
y=np.array(y)

"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)

x=[]
y=[]

#x_train=train_data[:7000,0]
#y_train=train_data[:7000,1]
#x_test=train_data[7000:,0]
#y_test=train_data[7000:,1]

#for i in range(len(x_train)):
#    x_train[i]=x_train[i].reshape(-1,120,160,3)
#    y_train[i]=np.array(y_train[i]).reshape(-1,3)
#    
#test_data = ImageDataGenerator()
#test_data = test_data.flow(x_train, y_train,batch_size=32)

def data_gen():
    while 1:
        for i in range(x_train.shape[0]):
            x=x_train[i].reshape(-1,120,160,3)
            y=np.array(y_train[i]).reshape(-1,3)
            yield (x, y)

def val_gen():
    while 1:
        for i in range(x_test.shape[0]):
            x=x_test[i].reshape(-1,120,160,3)
            y=np.array(y_test[i]).reshape(-1,3)
            yield (x, y)

checkpoint = ModelCheckpoint(filepath='/output/model_v3.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

classifier.fit_generator(data_gen(),
                         steps_per_epoch = x_train.shape[0],
                         epochs = 3,
                         validation_data = val_gen(),
                         validation_steps = x_test.shape[0],
                         callbacks=[checkpoint])
"""
classifier.fit(x,y, epochs=4, batch_size=32, verbose=1, validation_split=0.1)
classifier.save_weights('/output/model_opt.model')