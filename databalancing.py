# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 18:13:06 2017

@author: Deepak
"""
import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

for i in range(52):
    print('Processing file: ',i)
    
    lefts = []
    rights = []
    forwards = []
    
    file_name = 'data2/training_data-{}.npy'.format(i)
    train_data=np.load(file_name)
    
    shuffle(train_data)
    
    for data in train_data:
            
        img = data[0]
        choice = data[1]
        
        if choice  ==  [0,1,0,0,0,0]:
            forwards.append([img,[0,1,0]])
        elif choice == [0,0,0,1,0,0]:
            lefts.append([img,[1,0,0]])
        elif choice == [0,0,0,0,1,0]:
            rights.append([img,[0,0,1]])
    
    forwards = forwards[:len(lefts)][:len(rights)]
    lefts = lefts[:len(forwards)]
    rights = rights[:len(forwards)]

    final_data = forwards + lefts + rights
    shuffle(final_data)

    np.save(file_name, final_data)


#key_data=[]
#for i in range(52):
#        file_name = 'data2/training_data-{}.npy'.format(i)
#        train_data=np.load(file_name)
#        for data in train_data:
#            img=data[0]
#            choice=data[1]
#            key_data.append([choice])
##            print(choice)
##            cv2.imshow('test',img)
##            if cv2.waitKey(25) & 0xFF == ord('q'):
##                cv2.destroyAllWindows()
##                break
##        cv2.destroyAllWindows()
#
#df = pd.DataFrame(key_data)
#counts=Counter(df[0].apply(str))
#print(df.head)
#print(counts)
    
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()