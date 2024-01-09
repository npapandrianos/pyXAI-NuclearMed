# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:51:50 2021

@author: John
"""


''' LOAD BASIC LIBRARIRES'''

import matplotlib as plt
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from imutils import paths
import numpy as np
import random
import cv2
import os
from PIL import Image 
import numpy
import tensorflow as tf
import pandas as pd
from scipy.ndimage import rotate
SEED = 2   # set random seed


def normalize_from_pixels (scan):
    
    MIN_BOUND = scan.min()
    MAX_BOUND = scan.max()
    
    scan = (scan - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    scan[scan>1] = 1.
    scan[scan<0] = 0.
    return scan

def rotate_img(img, angle, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img


def gaussian_noise(img, mean=0, sigma=0.03):
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img+noise >= 1.0
    mask_overflow_lower = img+noise < 0
    noise[mask_overflow_upper] = 1.0
    noise[mask_overflow_lower] = 0
    img += noise
    return img
#imgplot = plt.imshow(img)

def load_parathyroid (path, excel_path, in_shape, verbose,label_col):
    
    #LOADS 3IN1 IMAGES (DUAL + SUBSTRACTION). NO PERIOXES
    
    WS = pd.read_excel(excel_path)
    excel = np.array(WS)
    excel[:,0] = excel[:,0].astype(int)
    
    info = np.empty([0,6])
    
    
    width = in_shape[0]
    height = in_shape[1]
    if verbose:
        print("[INFO] loading images")
        
        
    data_early = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths_early = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths_early) # Shuffle the image data
    
    # loop over the input images
    for l,imagePath_early in enumerate(imagePaths_early): #load, resize, normalize, etc
        if verbose:
            print("Preparing Image: {}".format(imagePath_early))
            
        image = cv2.imread(imagePath_early)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # FITLERS
        #ret,image =  cv2.threshold(image,100,255,cv2.THRESH_BINARY)
        #ret,image =  cv2.threshold(image,100,255,cv2.THRESH_TRUNC)
        
        #image = cv2.resize(image, (width, height))
        image = normalize_from_pixels (image)
        #image = cv2.resize(image, (width, height))

        
        # extract the class label from the image path and update the labels list
        img_num = imagePath_early.split(os.path.sep)[-1]
        img_num = img_num[:3]
        img_num = int(img_num)
        print(img_num)
        if verbose:
            print(img_num)
        for j in range(len(excel)):
            if int(excel[j,0]) == img_num:
                subject_no = j
                try:
                    label = int(excel[j,label_col])
                except Exception as e:
                    print(e)
                    label = 'else'
                if label == 1:
                    label2 = 'Parathyroid'
                    if verbose:
                        print ("IMG_NUM: {}, EXCEL LABEL: {}, Translated: {}".format(img_num,label,label2))
                elif label == 0:
                    label2 = 'ND'
                    if verbose:
                        print ("IMG_NUM: {}, EXCEL LABEL: {}, Translated: {}".format(img_num,label,label2))
            

        info = np.concatenate([info,excel[subject_no,:].reshape(1,6)])
        data_early.append(image)
        labels.append(label2)
            
    data_early = np.array(data_early, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    #labels = tf.keras.utils.to_categorical(labels, num_classes=2)
    print("Data and labels loaded and returned")
    return data_early, labels, labeltemp, image, image.max(), info






