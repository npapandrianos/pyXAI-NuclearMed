# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:10:21 2022

@author: User
"""

from volumentations import *
# https://github.com/ZFTurbo/volumentations

import matplotlib.pyplot as plt
import numpy as np
from random import *
import copy

def get_augmentation(patch_size,degrees,noise):
    
    #degrees = 20
    #noise = 0.02
    
    return Compose([
        Rotate((-degrees, degrees), (0, 0), (0, 0), p=1)], p=1.0)



def emerald_4d_augmentation (train_stress_ac,train_stress_nac,train_rest_ac,train_rest_nac,labels,OPTIONS_PREPROCESSING,nofaug = 3):
    
    
    '''
    This is an augmentation function that receives a train batch of size (number_of_elements,number_of_slices,width,height,channels)
    and augments element-wise and slice-wise.
    
    It also receives the labels and constructs the new label data.
    
    Specify the number of augmentations to happen.
    '''
    #####################################################
    # data must be list of np.arrays
    # labels must be np.array of equal size
    # OPTIONS must be dictionary with key 'shape' and value tuple (10, 64, 64,1): slices,w,h,channels)
    #####################################################
    
    shape = OPTIONS_PREPROCESSING['shape']
    
    aug_rest_ac = copy.deepcopy(train_rest_ac)
    aug_stress_ac = copy.deepcopy(train_stress_ac)
    aug_rest_nac = copy.deepcopy(train_rest_nac)
    aug_stress_nac = copy.deepcopy(train_stress_nac)

    ###############################################################
    # WE ASSIGN RANDOM ROTATION DEGREES AND NOISE LEVELS
    ##############################################################
    degrees = [randint(0,12) for _ in range (nofaug)]
    noise = [round(uniform(0.02, 0.05),5) for _ in range (nofaug)]    
    
    # number of augmented images to insert
    for iteration in range(nofaug):
    
        aug = get_augmentation(shape,degrees[iteration],noise[iteration]) #size must be tuple (10, 64, 64,1)      
        
        # for each instant
        for serie_num in range(len(train_rest_ac)):
            the_serie = train_rest_ac[serie_num,:,:,:,:]
            #plt.imshow(the_serie[5,:,:,:])
            
            # slice-wise augmentation
            aug_ser_1 = aug(**{'image': the_serie})['image']
            #plt.imshow(aug_ser_1[5,:,:,:])   
            
            aug_ser_1 = np.expand_dims(aug_ser_1, axis=0)
            aug_rest_ac = np.concatenate([aug_rest_ac,aug_ser_1],axis=0)

            
        for serie_num in range(len(train_rest_nac)):
            the_serie = train_rest_nac[serie_num,:,:,:,:]
            #plt.imshow(the_serie[5,:,:,:])
            
            # slice-wise augmentation
            aug_ser_1 = aug(**{'image': the_serie})['image']
            #plt.imshow(aug_ser_1[5,:,:,:])   
            
            aug_ser_1 = np.expand_dims(aug_ser_1, axis=0)
            aug_rest_nac = np.concatenate([aug_rest_nac,aug_ser_1],axis=0)        

        
        for serie_num in range(len(train_stress_ac)):
            the_serie = train_stress_ac[serie_num,:,:,:,:]
            #plt.imshow(the_serie[5,:,:,:])
            
            # slice-wise augmentation
            aug_ser_1 = aug(**{'image': the_serie})['image']
            #plt.imshow(aug_ser_1[5,:,:,:])   
            
            aug_ser_1 = np.expand_dims(aug_ser_1, axis=0)
            aug_stress_ac = np.concatenate([aug_stress_ac,aug_ser_1],axis=0)        
        
        
        for serie_num in range(len(train_stress_nac)):
            the_serie = train_rest_ac[serie_num,:,:,:,:]
            #plt.imshow(the_serie[5,:,:,:])
            
            # slice-wise augmentation
            aug_ser_1 = aug(**{'image': the_serie})['image']
            #plt.imshow(aug_ser_1[5,:,:,:])   
            
            aug_ser_1 = np.expand_dims(aug_ser_1, axis=0)
            aug_stress_nac = np.concatenate([aug_stress_nac,aug_ser_1],axis=0)        
        
            ## IN THE LAST ITERATION ADD THE LABEL OF THE INSTANCE
            labels = np.concatenate([labels,labels[serie_num,:].reshape(-1,2)])   
            
    return     aug_rest_ac,aug_rest_nac,aug_stress_ac,aug_stress_nac,labels
                
                
                
                
                
        
        
    

