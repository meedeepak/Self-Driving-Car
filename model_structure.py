# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:50:27 2017

@author: Deepak
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers

def create_model():
    classifier = Sequential()
    
    classifier.add(Conv2D(256,(3,3),input_shape=(120,160,3),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    
    classifier.add(Conv2D(256,(3,3),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    
    classifier.add(Conv2D(384,(3,3),activation='relu'))
    classifier.add(Conv2D(384,(3,3),activation='relu'))
    classifier.add(Conv2D(384,(3,3),activation='relu'))
    
    classifier.add(Conv2D(256,(3,3),activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    
    classifier.add(Flatten())
    
    classifier.add(Dense(units = 4096, activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 4096, activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 4096, activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units = 3, activation = 'softmax'))
    
    sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
    classifier.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return classifier