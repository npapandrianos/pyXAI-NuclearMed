# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:24:01 2022

@author: John
"""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, MaxPooling3D, Dropout, Conv3D, Input, GlobalAveragePooling3D, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import Model



def make_3d_main_70_emerald_exp (OPTIONS_MODE,OPTIONS_PREPROCESSING,OPTIONS_TRAINING):
    ###############################################################
    #  Achieved 76% on Experts, __% on MPI, __% on ICA
    ###############################################################
    def emerald_conv_3d_tw (x,filters,window,batch_norm,dropout,max_pool_yes,max_pool_window = (1,1,1),name=''):
        
        
        reg = tf.keras.regularizers.L2(l2=1e-4)
        
        if name == '':
            x  = Conv3D(filters, window, padding='valid', activation='relu',kernel_regularizer=reg)(x) # kernel_regularizer=reg
        else:
            x  = Conv3D(filters, window, padding='valid', activation='relu',kernel_regularizer=reg,name=name)(x)    
    
        
        if max_pool_yes:
           x = MaxPooling3D(max_pool_window,padding='valid')(x)
           
        if dropout:
           x = Dropout(0.1)(x)
           
        if batch_norm:
           x = BatchNormalization()(x)
           
        return x
    
    def make_block_emerald (inp,last_name):
        
        x = emerald_conv_3d_tw (inp,filters=8,window=(3,3,3),batch_norm=True,dropout=False,max_pool_yes=True,max_pool_window = (1,2,2),name='')
        x = emerald_conv_3d_tw (x,filters=16,window=(3,3,3),batch_norm=True,dropout=False,max_pool_yes=True,max_pool_window = (1,2,2),name='')
        
        x = emerald_conv_3d_tw (x,filters=32,window=(3,3,3),batch_norm=True,dropout=False,max_pool_yes=True,max_pool_window = (1,2,2),name='')
        x = emerald_conv_3d_tw (x,filters=64,window=(3,3,3),batch_norm=True,dropout=False,max_pool_yes=True,max_pool_window = (1,2,2),name=last_name)
    
        return x     
    
    
    
    classes = OPTIONS_TRAINING['classes']
    
    str_ac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])  
    str_ac = make_block_emerald (str_ac_in,last_name = 'str_ac')
    str_ac= tf.keras.layers.Flatten()(str_ac)
    #str_ac = GlobalAveragePooling3D()(str_ac)


    str_nac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])    
    str_nac = make_block_emerald (str_nac_in,last_name = 'str_nac')     
    str_nac= tf.keras.layers.Flatten()(str_nac)
    #str_nac= GlobalAveragePooling3D()(str_nac)

    
    res_nac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    res_nac = make_block_emerald (res_nac_in,last_name = 'rest_nac')     
    res_nac= tf.keras.layers.Flatten()(res_nac)
    #res_nac= GlobalAveragePooling3D()(res_nac)

    
    res_ac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    res_ac = make_block_emerald (res_ac_in,last_name = 'rest_ac') 
    res_ac= tf.keras.layers.Flatten()(res_ac)
    #res_ac= GlobalAveragePooling3D()(res_ac)

    
    
    n = tf.keras.layers.concatenate([str_ac,str_nac,res_ac,res_nac])
    n = Dense(600, activation='relu')(n)
    n = Dropout(0.4)(n)
    n = Dense(300, activation='relu')(n)
    n = Dropout(0.4)(n)
    n = Dense(classes, activation='softmax')(n)
    
    model = Model(inputs=[str_ac_in,str_nac_in,res_ac_in,res_nac_in], outputs=n) 
    opt = 'Adam'
    ''' COMPILATION'''
    model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # model.summary()
    return model 




###########################################################################
#
#
#
# STABLE 2ND RELEASE (MODERN FUNCTIONS, NOTHING ELSE)
#
#
#
###########################################################################



def make_3d_main_70_emerald (OPTIONS_MODE,OPTIONS_PREPROCESSING,OPTIONS_TRAINING):
    ###############################################################
    #  Achieved 76% on Experts, __% on MPI, __% on ICA
    ###############################################################
    def emerald_conv_3d_tw (x,filters,window,batch_norm,dropout,max_pool_yes,max_pool_window = (1,1,1),name=''):
        
        
        reg = tf.keras.regularizers.L2(l2=1e-4)
        
        if name == '':
            x  = Conv3D(filters, window, padding='valid', activation='relu',kernel_regularizer=reg)(x) # kernel_regularizer=reg
        else:
            x  = Conv3D(filters, window, padding='valid', activation='relu',kernel_regularizer=reg,name=name)(x)    
    
        
        if max_pool_yes:
           x = MaxPooling3D(max_pool_window,padding='valid')(x)
           
        if dropout:
           x = Dropout(0.1)(x)
           
        if batch_norm:
           x = BatchNormalization()(x)
           
        return x
    
    def make_block_emerald (inp,last_name):
        
        x = emerald_conv_3d_tw (inp,filters=8,window=(3,3,3),batch_norm=True,dropout=False,max_pool_yes=True,max_pool_window = (1,2,2),name='')
        x = emerald_conv_3d_tw (x,filters=16,window=(3,3,3),batch_norm=True,dropout=False,max_pool_yes=True,max_pool_window = (1,2,2),name='')
        
        x = emerald_conv_3d_tw (x,filters=32,window=(3,3,3),batch_norm=True,dropout=False,max_pool_yes=True,max_pool_window = (1,2,2),name='')
        x = emerald_conv_3d_tw (x,filters=64,window=(3,3,3),batch_norm=True,dropout=False,max_pool_yes=True,max_pool_window = (1,2,2),name='')
    
        return x     
    
    
    
    classes = OPTIONS_TRAINING['classes']
    
    str_ac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])  
    str_ac = make_block_emerald (str_ac_in,last_name = 'str_ac')
    str_ac= tf.keras.layers.Flatten()(str_ac)
    #str_ac = GlobalAveragePooling3D()(str_ac)


    str_nac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])    
    str_nac = make_block_emerald (str_nac_in,last_name = 'str_nac')     
    str_nac= tf.keras.layers.Flatten()(str_nac)
    #str_nac= GlobalAveragePooling3D()(str_nac)

    
    res_nac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    res_nac = make_block_emerald (res_nac_in,last_name = 'rest_nac')     
    res_nac= tf.keras.layers.Flatten()(res_nac)
    #res_nac= GlobalAveragePooling3D()(res_nac)

    
    res_ac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    res_ac = make_block_emerald (res_ac_in,last_name = 'rest_ac') 
    res_ac= tf.keras.layers.Flatten()(res_ac)
    #res_ac= GlobalAveragePooling3D()(res_ac)

    
    
    n = tf.keras.layers.concatenate([str_ac,str_nac,res_ac,res_nac])
    n = Dense(600, activation='relu')(n)
    n = Dropout(0.4)(n)
    n = Dense(300, activation='relu')(n)
    n = Dropout(0.4)(n)
    n = Dense(classes, activation='softmax')(n)
    
    model = Model(inputs=[str_ac_in,str_nac_in,res_ac_in,res_nac_in], outputs=n) 
    opt = 'Adam'
    ''' COMPILATION'''
    model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # model.summary()
    return model 






###########################################################################
#
#
#
# STABLE 1ST RELEASE
#
#
#
###########################################################################


    
def make_3d_main_70 (OPTIONS_MODE,OPTIONS_PREPROCESSING,OPTIONS_TRAINING):
    ###############################################################
    #  Achieved 76% on Experts, __% on MPI, __% on ICA
    ###############################################################
    
    def conv3D (x,filters,bn,maxp=0,rdmax=0,drop=True,DepthPool=False,name=''):
        if name == '':
            x  = Conv3D(filters, (3,3,3), padding='valid', activation='relu')(x)
        else:
            x  = Conv3D(filters, (3,3,3), padding='valid', activation='relu',name=name)(x)
        
        
        if maxp ==1 and DepthPool==False:
            x = MaxPooling3D((1,2, 2),padding='valid')(x)
        if  DepthPool==True:   
            x = MaxPooling3D((2,2, 2),padding='valid')(x)
            
        if rdmax == 1:
            x = MaxPooling3D((2,2, 2),padding='valid')(x)
        if drop==True:
            x = Dropout(0.4)(x)
            
        if bn==1:
            x = BatchNormalization()(x)   
        return x    

    classes = OPTIONS_TRAINING['classes']
    
    str_ac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    str_ac = conv3D (str_ac_in,filters=8,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='')
    str_ac = conv3D (str_ac,16,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='')
    str_ac = conv3D (str_ac,32,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='')
    str_ac = conv3D (str_ac,64,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='str_ac')
    #str_ac = conv3D (str_ac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #str_ac = conv3D (str_ac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #str_ac = GlobalAveragePooling3D()(str_ac)
    str_ac= tf.keras.layers.Flatten()(str_ac)

    str_nac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    str_nac = conv3D (str_nac_in,filters=8,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='')
    str_nac = conv3D (str_nac,16,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='')
    str_nac = conv3D (str_nac,32,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='')
    str_nac = conv3D (str_nac,64,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='str_nac')
    #str_nac = conv3D (str_nac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #str_nac = conv3D (str_nac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #str_nac = GlobalAveragePooling3D()(str_nac)      
    str_nac= tf.keras.layers.Flatten()(str_nac)

    res_nac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    res_nac = conv3D (res_nac_in,filters=8,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='')
    res_nac = conv3D (res_nac,16,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='')
    res_nac = conv3D (res_nac,32,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='')
    res_nac = conv3D (res_nac,64,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='rest_nac')
    #res_nac = conv3D (res_nac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #res_nac = conv3D (res_nac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #res_nac = GlobalAveragePooling3D()(res_nac)
    res_nac= tf.keras.layers.Flatten()(res_nac)
    
    res_ac_in = Input(shape=OPTIONS_PREPROCESSING['shape'])
    res_ac = conv3D (res_ac_in,filters=8,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='')
    res_ac = conv3D (res_ac,16,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='')
    res_ac = conv3D (res_ac,32,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='')
    res_ac = conv3D (res_ac,64,bn = 1, maxp=1,rdmax=0,drop=False, DepthPool=False,name='rest_ac')
    #res_ac = conv3D (res_ac,128,bn = 1, maxp=1,rdmax=0,drop=True, DepthPool=False)
    #res_ac = conv3D (res_ac,256,bn = 1, maxp=0,rdmax=0,drop=True, DepthPool=False)
    #res_ac = GlobalAveragePooling3D()(res_ac)
    res_ac= tf.keras.layers.Flatten()(res_ac)
        
    n = tf.keras.layers.concatenate([str_ac,str_nac,res_ac,res_nac], axis=-1)
    n = Dense(600, activation='relu')(n)
    n = Dropout(0.4)(n)
    n = Dense(300, activation='relu')(n)
    n = Dropout(0.4)(n)
    # n = Dense(4096, activation='selu')(c)
    # n = Dropout(0.5)(n)
    #n = Dense(750, activation='elu')(n)
    #n = Dropout(0.5)(n)
    n = Dense(classes, activation='softmax')(n)
    
    
    model = Model(inputs=[str_ac_in,str_nac_in,res_ac_in,res_nac_in], outputs=n)
    
    #opt = SGD(lr=0.01)
    
    opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
    
    opt = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False)
    
    
    model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # model.summary()
    return model










###########################################################################
#
#
#
# ALL IN ONE SCAN
#
#
#
###########################################################################







def make_volume_all (OPTIONS_MODE,OPTIONS_PREPROCESSING,OPTIONS_TRAINING):
    
    def emerald_conv_3d_tw (x,filters,window,batch_norm,dropout,max_pool_yes,max_pool_window = (1,1,1),name=''):
        
        
        reg = tf.keras.regularizers.L2(l2=1e-4)
        
        if name == '':
            x  = Conv3D(filters, window, padding='valid', activation='relu',kernel_regularizer=reg)(x) # kernel_regularizer=reg
        else:
            x  = Conv3D(filters, window, padding='valid', activation='relu',kernel_regularizer=reg,name=name)(x)    
    
        
        if max_pool_yes:
           x = MaxPooling3D(max_pool_window,padding='valid')(x)
           
        if dropout:
           x = Dropout(0.1)(x)
           
        if batch_norm:
           x = BatchNormalization()(x)
           
        return x
    
    
    def make_block_vol (inp,last_name):
        
        x = emerald_conv_3d_tw (inp,filters=64,window=(3,3,3),batch_norm=False,dropout=False,max_pool_yes=True,max_pool_window = (2,2,2),name='')
        x = emerald_conv_3d_tw (x,filters=64,window=(3,3,3),batch_norm=False,dropout=False,max_pool_yes=True,max_pool_window = (2,2,2),name='')
        
        x = emerald_conv_3d_tw (x,filters=128,window=(3,3,3),batch_norm=False,dropout=False,max_pool_yes=False,max_pool_window = (1,2,2),name='')
        x = emerald_conv_3d_tw (x,filters=128,window=(3,3,3),batch_norm=False,dropout=False,max_pool_yes=True,max_pool_window = (2,2,2),name='')
        
        x = emerald_conv_3d_tw (x,filters=256,window=(3,3,3),batch_norm=False,dropout=False,max_pool_yes=False,max_pool_window = (1,2,2),name='')
        x = emerald_conv_3d_tw (x,filters=256,window=(3,3,3),batch_norm=False,dropout=False,max_pool_yes=False,max_pool_window = (1,2,2),name='')
        x = emerald_conv_3d_tw (x,filters=256,window=(3,3,3),batch_norm=False,dropout=False,max_pool_yes=True,max_pool_window = (2,2,2),name='')
       
        
        #x = emerald_conv_3d (x,filters=256,window=(1,3,3),batch_norm=True,dropout=True,max_pool_yes=False,max_pool_window = (1,1,1),name='')
        #x = emerald_conv_3d (x,filters=256,window=(1,3,3),batch_norm=True,dropout=True,max_pool_yes=True,max_pool_window = (1,2,2),name='')
        #x = emerald_conv_3d (x,filters=256,window=(1,3,3),batch_norm=True,dropout=True,max_pool_yes=False,max_pool_window = (1,2,2),name='')
        #x = emerald_conv_3d (x,filters=512,window=(1,3,3),batch_norm=True,dropout=True,max_pool_yes=False,max_pool_window = (1,2,2),name=last_name)      
        
        return x     



    classes = OPTIONS_TRAINING['classes']
    shape = (OPTIONS_PREPROCESSING['shape'][0]*4,OPTIONS_PREPROCESSING['shape'][1],OPTIONS_PREPROCESSING['shape'][2],OPTIONS_PREPROCESSING['shape'][3])


    inp = Input(shape=shape)
    str_ac = make_block_vol (inp,last_name = 'str_ac')
    str_ac = GlobalAveragePooling3D()(str_ac)

    n = Dense(1024, activation='relu')(str_ac)
    n = Dropout(0.5)(n)
    n = Dense(512, activation='relu')(n)
    n = Dropout(0.5)(n)
    n = Dense(classes, activation='softmax')(n)
    n = Dense(classes, activation='softmax')(n)
    
    
    model = Model(inputs=inp, outputs=n)
    
    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    
    #opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
    
    opt = tf.keras.optimizers.Adam(
    learning_rate=lr_schedule,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False)
    
    
    model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.summary()
    return model    