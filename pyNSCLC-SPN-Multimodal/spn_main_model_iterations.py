# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:50:16 2021

@author: John
"""

'''CODES FOR PARATHYROID'''

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.config.list_physical_devices('GPU')


sys.path.insert(1, 'C:\\Users\\User\\DSS EXPERIMENTS\\Fruit Quality\\Pythons\\')

from fr_data_loader import load_fruit, load_fruit_ext,load_fruit_split
from fr_main_functions import train_multi,model_save_load


import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(os.getenv('‘TF_GPU_ALLOCATOR’'))

import logging

#%% PARAMETER ASSIGNEMENT

fruits = ['Apple','Banana','Cucumber','Grape','Guava',
          'Kaki','Lemon','Lime','Mango','Orange','Papaya',
          'Peach','Pear','Pepper','Pomegranate','Tomato','Watermelon']
path = 'D:\\DATASETS COLLECTION (ALL) - MAIN STORE FOLDER\\Fruit Quality Paper Data\\'

fruit = 'All'
path = os.path.join(path,fruit)



logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level = logging.DEBUG, filename = 'C:\\Users\\User\\DSS EXPERIMENTS\\Fruit Quality\\Pythons\\mainFruit_logger_{}.log'.format(fruit))
logging.captureWarnings(False)
logging.getLogger('matplotlib.font_manager').disabled = True




'''MULTI'''
in_shape  = (128,128,3) # MULTI SIAM
tune = 1 # SET: 1 FOR TRAINING SCRATCH, 0 FOR OFF THE SHELF, INTEGER FOR TRAINABLE LAYERS (FROM TUNE AND DOWN, THE LAYERS WILL BE TRAINABLE)
epochs = 80 #500 
batch_size = 4
n_split = 10 #set to 10
augmentation=False
verbose=True
class_names = ["Rotten","Fresh"] # 0 = Bad = Rotten || 1 = Good = Fresh . So the Positive Class is Fresh

# att_vgg19  lvgg    inception    vgg19_base    ffvgg19    att_ffvgg19  efficient 
# ioapi_vit ioapi_swimtr ioapi_perceiver ioapi_involutional ioapi_convmixer
#  ioapi_big_transfer ioapi_eanet ioapi_fnet ioapi_gmlp ioapi_mlpmixer


model = 'ioapi_mlpmixer' 
classes = 2

#%%
##%% IMAGE LOAD

# for fruits
data_all, labels_all, image = load_fruit (path, in_shape, verbose=False)

# for All
data_all, labels_all, image,data_train,labels_train,data_test,labels_test = load_fruit_split (path, in_shape, verbose)


data = data_all
labels = labels_all


imgs_index = [0,10,101,201]

# for number in imgs_index:
#     image = data[number,:,:,:]
#     plt.imshow(image)
#     plt.show()


fresh = int(np.sum(labels_all,axis=0)[1])
rotten = int(np.sum(labels_all,axis=0)[0])

print ('=======TRAIN SET======')
print ('Fresh: {}'.format(fresh))
print ('Rotten: {}'.format(rotten))

logging.info('=======TRAIN SET======')
logging.info('Fresh: {}'.format(fresh))
logging.info('Rotten: {}'.format(rotten))


path2 = 'D:\\DATASETS COLLECTION (ALL) - MAIN STORE FOLDER\\Fruit Quality Paper Data\\ALL_test'
path2 = os.path.join(path2,fruit)
data_all_exter, labels_all_exter, image_exter = load_fruit_ext (path2, in_shape, verbose=False)

#%% FIT THE MODEL TO THE DATA (FOR PHASE 2)

from fr_main_functions import model_save_load,train_multi
import time
''' TRAIN - EVALUATE MODEL - GET METRICS '''


# prepare labels according to models
from sklearn import preprocessing
in_shape_for_model = in_shape



testing_models1 = ['att_vgg19','lvgg','inception','vgg19_base','ffvgg19','att_ffvgg19']
testing_models2 = ['ioapi_vit','ioapi_swimtr','ioapi_perceiver','ioapi_involutional','ioapi_convmixer']
testing_models3 = ['ioapi_big_transfer','ioapi_eanet','ioapi_fnet','ioapi_gmlp','ioapi_mlpmixer']

# att_vgg19  lvgg    inception    vgg19_base    ffvgg19    att_ffvgg19  efficient 
# ioapi_vit ioapi_swimtr ioapi_perceiver ioapi_involutional ioapi_convmixer
#  ioapi_big_transfer ioapi_eanet ioapi_fnet ioapi_gmlp ioapi_mlpmixer

testing_models = ['vgg19_base']

for model in testing_models:
    


    if model in ['ioapi_vit','ioapi_perceiver','ioapi_fnet','ioapi_gmlp','ioapi_mlpmixer', 
                 'ioapi_involutional','ioapi_convmixer','ioapi_big_transfer']:
        labelsT = labels_all
        labelsT = labels_all[:,1]
    else:
        labelsT = labels_all
    
    
    
    
    
    # 10-FOLD
    # start = time.time()
    
    # model3, group_results, fold_scores, pd_metrics,predictions_all,predictions_all_num,test_labels,labels,conf_final,cnf_matrix,conf_img,history = train_multi(data,labels=labelsT,epochs=epochs,batch_size=batch_size, model=model, in_shape=in_shape_for_model, tune=tune, classes=classes,n_split=n_split,augmentation=augmentation,verbose=verbose,logs=True,plot_results=True,class_names=class_names,save_variables=True)
    
    # stop = time.time()
    # print ('Elapsed for k-fold: {} seconds'.format(stop-start))
    # logging.info('Elapsed for k-fold: {} seconds'.format(stop-start))
    
    
    
    # SAVE ONLY
    start = time.time()
    
    model3, group_results, fold_scores, pd_metrics,predictions_all,predictions_all_num,test_labels,labels,conf_final,cnf_matrix,conf_img,history = train_multi(data,labels=labelsT,epochs=epochs,batch_size=batch_size, model=model, in_shape=in_shape_for_model, tune=tune, classes=classes,n_split=n_split,augmentation=augmentation,verbose=verbose,logs=True,plot_results=True,class_names=class_names,save_variables=True,save_model_just=True)
    
    stop = time.time()
    print ('Elapsed for just train once: {} seconds'.format(stop-start))
    logging.info('Elapsed for just train once: {} seconds'.format(stop-start))
    
    
    
    fresh = int(np.sum(labels_all_exter,axis=0)[1])
    rotten = int(np.sum(labels_all_exter,axis=0)[0])
    
    print ('=======TEST SET======')
    print ('Fresh: {}'.format(fresh))
    print ('Rotten: {}'.format(rotten))
    
    logging.info('=======TEST SET======')
    logging.info('Fresh: {}'.format(fresh))
    logging.info('Rotten: {}'.format(rotten))
    
    
    start = time.time()
    with tf.device('/CPU:0'):
        predictions_all_catTEST = model3.predict(data_all_exter)
        predictions_all_numTEST = model3.predict(data_all_exter) #for def models functional api
        predictions_allTEST = np.argmax(predictions_all_catTEST, axis=-1)
        
        
        import pandas as pd
        pd.DataFrame(predictions_allTEST).to_csv("predictions_fruit.csv")
        pd.DataFrame(labels_all_exter).to_csv("labels_fruit.csv")
        
        import fr_metrics
        
        THE_METRICS = fr_metrics.metrics (predictions_allTEST, predictions_all_numTEST, predictions_all_catTEST, labels_all_exter, verbose = True)
    
        logging.info('Test set got the metrics as follows:')
        logging.info(THE_METRICS)
    
    stop = time.time()
    print ('Elapsed for predicting new instances: {} seconds'.format(stop-start))
    logging.info('Elapsed for predicting new instances: {} seconds'.format(stop-start))
    
    
    #model3.save('{}_{}_model'.format(fruit,model))
    #new_model = tf.keras.models.load_model('{}_{}_model'.format(fruit,model))



#%%

'''
LIME
'''

# import mr_lime_func

# # LIME COMMANDS
# #items_no = [17,18, 100, 500, 600, 601, 602, 603, 604, 152]
# items_no = [i for i in range (len(data[:30]))]
# base_path = 'C:\\Users\\User\\DSS EXPERIMENTS\\MRI Classification - Explainability\\XAI\\GLIOMA\\'

# mr_lime_func.the_lime (items_no,predictions_all,labels,data,1,model3,verbose = False,show=False, save = True, base_path=base_path)



# #%%


# '''

# GRAD-CAM PLUS PLUS


# '''

# import mr_gradcamplusplus

# items_no = [i for i in range (len(data[:50]))]
# #items_no = [17,18, 100, 500, 600, 601, 602, 603, 604, 152]
# base_path = 'C:\\Users\\User\\DSS EXPERIMENTS\\MRI Classification - Explainability\\XAI\\GLIOMA\\'


# # GradCAM++
# mr_gradcamplusplus.gradcamplusplus (items_no,predictions_all,labels,data,1,model3,verbose = False,show=False, save = True, base_path=base_path)


# # Score CAM
# mr_gradcamplusplus.scorecam (items_no,predictions_all,labels,data,1,model3,verbose = False,show=False, save = True, base_path=base_path)

# # GradCAM
# mr_gradcamplusplus.gradcam (items_no,predictions_all,labels,data,1,model3,verbose = False,show=False, save = True, base_path=base_path)


# # Saliency
# mr_gradcamplusplus.saliency (items_no,predictions_all,labels,data,1,model3,verbose = False,show=False, save = True, base_path=base_path)

# # Smooth Grad
# mr_gradcamplusplus.smoothgrad (items_no,predictions_all,labels,data,1,model3,verbose = False,show=False, save = True, base_path=base_path)













