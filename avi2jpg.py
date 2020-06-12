# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:39:50 2020

@author: zhijiezheng
"""

import cv2

def avi_to_jpg(video,train,test):
    frame = 0
    cap = cv2.VideoCapture(video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    factor = 2
    while True:
        check ,img = cap.read()
        if check:
            if frame < 4500:
                path = train
            else:
                path = test
            img = cv2.resize(img,(1920//factor,1080//factor))
            cv2.imwrite(path+"//"+str(frame)+".jpg",img)
            frame+=1
        else:
            break
    cap.release()
    
if __name__ == '__main__':
    avi_to_jpg("TownCentreXVID.avi",train='./train_images',test='./test_images')