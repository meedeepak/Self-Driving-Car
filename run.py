# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:53:21 2017

@author: Deepak
"""

import numpy as np
import time
from grabscreen import grab_screen
import cv2
from getkeys import key_check
from directKeys import PressKey,ReleaseKey, W, A, S, D
from model_structure import create_model

def straight():
    ReleaseKey(A)
    ReleaseKey(D)
    PressKey(W)

def left():
    ReleaseKey(D)
    PressKey(W)
    PressKey(A)

def right():
    ReleaseKey(A)
    PressKey(W)
    PressKey(D)

def main():
    model=create_model()
    model.load_weights('model_opt.hdf')

#    weights = model.layers[0].get_weights()[0]
#    biases = model.layers[0].get_weights()[1]
    
    for i in list(range(5))[::-1]:
        print(i+1)
        time.sleep(1)
        
    paused = False
    while(True):
        
        if not paused:
            screen = grab_screen(region=(0,64,640,480))
            screen = cv2.resize(screen, (160,120))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            screen = np.expand_dims(screen, axis = 0)
            prediction = model.predict(screen)[0]
#            print(prediction)
            
            maxval=max(prediction)

            if prediction[1] == maxval:
                straight()
                print('straigh',maxval)
            elif prediction[2] == maxval:
                right()
                print('right',maxval)
            elif prediction[0] == maxval:
                left()
                print('left',maxval)

        keys = key_check()

        if 'P' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()