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


sys.path.insert(1, 'C:\\Users\\User\\DSS EXPERIMENTS\\ZZZ. Parathyroid 2\\\Pythons\\1img-1net-grad-lime')

# sys.path.insert(1, 'C:\\Users\\japostol\\Downloads\\Para')

from p2_data_loader import load_parathyroid
from p2_main_functions import train_multi,model_save_load,feature_maps



#%% PARAMETER ASSIGNEMENT


'''MULTI NO FACE'''
path = 'C:\\Users\\User\\DSS EXPERIMENTS\\ZZZ. Parathyroid 2\\DATA\\3in1_cropped\\'
excel_path = 'C:\\Users\\User\\DSS EXPERIMENTS\\ZZZ. Parathyroid 2\\DATA\\labels_full.xlsx'



'''MULTI'''
in_shape2  = (555,185,3) # MULTI SIAM
tune = 1 # SET: 1 FOR TRAINING SCRATCH, 0 FOR OFF THE SHELF, INTEGER FOR TRAINABLE LAYERS (FROM TUNE AND DOWN, THE LAYERS WILL BE TRAINABLE)
epochs = 200 #set to 200
batch_size = 40
n_split = 10 #set to 10
augmentation=True
verbose=True
class_names = ["Healhty", "Parathyroid"]
model = 'lvgg'
#%%
##%% IMAGE LOAD

in_shape2 = (555,185,3) # those dimension aspects are correct !!!!DO NOT CHANGE!!!
label_column = 4 # 2 FOR DUAL PHASE. 3 FOR SUB PHASE. 4 FOR BOTH
data, labels, labeltemp, image, image_max, info = load_parathyroid (path, excel_path, in_shape2, verbose=True,label_col=label_column)


imgs_index = [0,100,200,303,308,500]

for number in imgs_index:
    image = data[number,:,:,:]
    plt.imshow(image)
    plt.show()


#%% FIT THE MODEL TO THE DATA (FOR PHASE 2)

from p2_main_functions import model_save_load,feature_maps,train_multi
''' TRAIN - EVALUATE MODEL - GET METRICS '''


# double input
#in_shape_model = (in_shape2[0],in_shape2[1],in_shape2[2])



'''3 VGGS'''
from sklearn import preprocessing
labels = tf.keras.utils.to_categorical(labels, num_classes=2)
in_shape2  = (555,185,3)
classes = 2


# 10-FOLD

model3, group_results, fold_scores, pd_metrics,predictions_all,predictions_all_num,test_labels,labels,conf_final,cnf_matrix,conf_img,history = train_multi(data,labels=labels,epochs=epochs,batch_size=batch_size, model=model, in_shape=in_shape2, tune=tune, classes=classes,n_split=n_split,augmentation=augmentation,verbose=verbose,logs=True,plot_results=False,class_names=class_names,save_variables=True)

# SAVE ONLY
model3, group_results, fold_scores, pd_metrics,predictions_all,predictions_all_num,test_labels,labels,conf_final,cnf_matrix,conf_img,history = train_multi(data,labels=labels,epochs=epochs,batch_size=batch_size, model=model, in_shape=in_shape2, tune=tune, classes=classes,n_split=n_split,augmentation=augmentation,verbose=verbose,logs=True,plot_results=False,class_names=class_names,save_variables=True,save_model_just=True)




predictions_all = model3.predict(data) #for def models functional api
predictions_all = np.argmax(predictions_all, axis=-1)


import pandas as pd
pd.DataFrame(info).to_csv("file.csv")
pd.DataFrame(predictions_all).to_csv("predictions.csv")
pd.DataFrame(labels).to_csv("labels.csv")



#%% GRad CAM

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm



# to load an external image
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array1, model3, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model3.inputs], [model3.get_layer(last_conv_layer_name).output, model3.output]
    )
    
    #  grad_model = tf.keras.models.Model(
    #     [model3.inputs], [model3.get_layer('model').get_layer(last_conv_layer_name).output, model3.output]
    # )   

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array1)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# # Display heatmap
# plt.matshow(heatmap)
# plt.show()


def save_and_display_gradcam(img, heatmap, cam_path="no_name.jpg", alpha=0.4, save=True):
    # Load the original image
    #img = keras.preprocessing.image.load_img(img_path)
    #img = keras.preprocessing.image.img_to_array(img)
    # Rescale heatmap to a range 0-255
    heatmap2 = np.uint8(255 * heatmap)
    img  = img*255

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap2]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    if not save:
        default_path = "temp.jpg"
        # Save the superimposed image
        superimposed_img.save(default_path)
    
        # Display Grad CAM
        display(Image(default_path))
    else:
        superimposed_img.save(cam_path)
    
        # Display Grad CAM
        #display(Image(cam_path))        



def GRADCAM ( model3, img_list, info, data,labels, 
             last_conv_layer_name, alpha = 0.4, save = True ):
    
    '''
        model3: the saved model (tf file)
        img_lsit: a list of img numbers that corresponds to the data array. E.g [0,1,200]
        labels: the whole labels list
        last_conv_layer_name: found with mode3.layer_names
        alpha: defaults to 0.4
        save: True \ False
        base_name_save: the base name.The final files will be saved with "base_name_save_grad_x_label"
    
    '''
    

    # Remove last layer's softmax
    model3.layers[-1].activation = None
    
    for img_no in img_list:
    
        # display(Image(img_path))
        img_array1 = data[img_no,:,:,:]
        img_array1 = np.expand_dims(img_array1, axis=0)
 
        
        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array1, model3, last_conv_layer_name)
        
        
        if save:
            
            if labels[img_no][0]==1:
                label = 'ISnormal'
            else:
                label = 'ISparathyroid'
                
            if predictions_all[img_no] == 1:
                predicted = 'predPARATHYR'
            else:
                predicted = 'predNORMAL'
            
            name = '{}_all_{}_{}.jpg'.format(info[img_no,0],label,predicted)
            save_and_display_gradcam(img = data[img_no,:,:,:], heatmap=heatmap,cam_path=name, alpha=alpha, save=True)
            
            img_array1 = tf.keras.preprocessing.image.array_to_img(img_array1[0,:,:,:])
            
            the_path = '{}_all.jpg'.format(info[img_no,0])
            
            img_array1.save(the_path)
            
            print ('Label: {}'.format(labels[img_no]))
            print ('Predicted: {}'.format(predictions_all[img_no]))
        else:
            name = 'temp.jpg'
            save_and_display_gradcam(img = data[img_no,:,:,:], heatmap=heatmap,cam_path=name, alpha=alpha, save=True)
            print ('Label: {}'.format(labels[img_no]))
            print ('Predicted: {}'.format(predictions_all[img_no]))





# DEFINE THE NAME OF THE LAST LAYER
for layer in model3.layers:
    print(layer.name)

last_conv_layer_name = "block5_conv3"



# DEFINE LIST OF IMAGES
img_list = [0,1,2,3,4,100,200,300,400,500,600]
img_list = [i for i in range (len(data))]

# CALL THE HYPER-FUNCTION
GRADCAM ( model3, img_list, info, data, labels, last_conv_layer_name, alpha = 0.4, save = True )


#%% LIME

from lime import lime_image
# https://lime-ml.readthedocs.io/en/latest/lime.html
explainer = lime_image.LimeImageExplainer(verbose=True)

explanation = explainer.explain_instance(data[5].astype('double'), model3.predict,  
                                         top_labels=2, hide_color=0, num_samples=1000, num_features = 100000)


from skimage.segmentation import mark_boundaries

temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True,negative_only = False, num_features=2, hide_rest=False)
temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False,negative_only = False, num_features=10, hide_rest=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))
ax1.imshow(mark_boundaries(temp_1, mask_1))
ax2.imshow(mark_boundaries(temp_2, mask_2))
ax1.axis('off')
ax2.axis('off')

