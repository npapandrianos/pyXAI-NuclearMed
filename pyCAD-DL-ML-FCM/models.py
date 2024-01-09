# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:09:38 2022

@author: John
"""

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, MaxPooling3D, Dropout, Conv3D, Input, GlobalAveragePooling3D, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import Model

import time

def make_vgg19 (OPTIONS_MODE,OPTIONS_PREPROCESSING,OPTIONS_TRAINING): #tune = 0 is off the self, tune = 1 is scratch, tune 
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # NOT READY FOR SERIES 

    classes = OPTIONS_TRAINING['classes']
    in_shape = OPTIONS_PREPROCESSING['shape']
    tune = OPTIONS_TRAINING['tune']
   
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    
    
    if tune == 'scratch':
        for layer in base_model.layers:
            layer.trainable = True
    elif tune == 'frozen':
        for layer in base_model.layers:
            layer.trainable = False
        
    else:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[20:]:
            layer.trainable = True

    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)

    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model



# def FCM ():
    
#     return predictions


# def rf ():
#     start = time.time()
    
    
    
    
#     end = time.time()
#     duration = (end - start)
#     return 