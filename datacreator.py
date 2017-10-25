import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
import os

a =  [1,0,0,0,0,0]
w =  [0,1,0,0,0,0]
d =  [0,0,1,0,0,0]
wa = [0,0,0,1,0,0]
wd = [0,0,0,0,1,0]
nk = [0,0,0,0,0,1]

def keys_to_output(keys):
    
    output = [0,0,0,0,0,0]

    if 'W' in keys and 'A' in keys:
        output = wa
    elif 'W' in keys and 'D' in keys:
        output = wd
    elif 'W' in keys:
        output = w
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = nk
        
    return output

def main():
    
    starting_value=0
    training_data = []
    while True:
        file_name = 'data2/training_data-{}.npy'.format(starting_value)
        if os.path.isfile(file_name):
            print('File exists, moving along',starting_value)
            starting_value += 1
        else:
            print('File does not exist, starting fresh!',starting_value)
            break
    
    for i in list(range(4)) [::-1]:
        print(i+1)
        time.sleep(1)
    
    paused = False
    
    print('STARTING!!!')    
    while(True):
        if not paused:
            screen =  grab_screen(region=(0,64,640,480))
            screen = cv2.resize(screen,(160,120))
            screen = cv2.cvtColor(screen,cv2.COLOR_BGR2RGB)
            
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen,output])
#            cv2.imshow('test',screen)
#            if cv2.waitKey(25) & 0xFF == ord('q'):
#                cv2.destroyAllWindows()
#                break
#            print(output)

            if len(training_data) % 500 == 0:
                 print(len(training_data))
                 if len(training_data) == 1000:
                     np.save(file_name,training_data)
                     training_data = []
                     file_name = 'data2/training_data-{}.npy'.format(starting_value)
                     starting_value += 1
                
        keys = key_check()
        if 'P' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
#                cv2.destroyAllWindows()
                print('Paused!')
                paused = True
                time.sleep(1)           
main()