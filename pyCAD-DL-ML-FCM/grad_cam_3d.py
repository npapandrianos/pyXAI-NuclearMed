#
# https://github.com/fitushar/3D-GuidedGradCAM-for-Medical-Imaging
# ADAPTED FROM THE ABOVE

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import datetime
import numpy as np
import pandas as pd
import math
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from skimage.transform import resize
from scipy.ndimage import zoom


#
def plot_slices(num_rows, num_columns, width, height, data,save_path,save_name,grad=False):
    """Plot a montage of 20 CT slices"""

    data = np.swapaxes(data,0,3) #swap slice number dimension with the channle dimension
    data = data[0,:,:,:] # destroy the first dimension
    
    data = np.rot90(np.array(data))
    data = np.transpose(data)

    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 18.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    #plt.show()
    
    save_name = save_name+'.png'
    complete = os.path.join(save_path,save_name)
    plt.savefig(complete)
    plt.close()


# data = fused_str_ac

def plot_slices_grad(num_rows, num_columns, width, height, data,save_path,save_name):
    """Plot a montage of 20 CT slices"""
    from matplotlib import pyplot as plt
    data = np.swapaxes(data,0,3) #swap slice number dimension with the channle dimension

    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height,3))

    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 18.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j])
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    #plt.show()
    
    save_name = save_name+'.png'
    complete = os.path.join(save_path,save_name)
    plt.savefig(complete)
    plt.close()

# Function to get the image chunk fot guided GradCAM
def Get_image_array_Array_and_give_chunk(image_array,patch_slice_slice):

    Devide_integer=image_array.shape[0] // patch_slice_slice
    Reminder= image_array.shape[0] % patch_slice_slice
    print('CT Volume_Shape={}'.format(image_array.shape))
    print('Devide_integer={}'.format(Devide_integer))
    print('Reminder={}'.format(Reminder))
    print('Total of {} + {} ={} Should ={}'.format(patch_slice_slice*Devide_integer,Reminder,patch_slice_slice*Devide_integer+Reminder,image_array.shape[0]))

    lastpatch_starts_from= (image_array.shape[0])-patch_slice_slice
    print(lastpatch_starts_from)

    patch_list=[]
    patch_start=0
    patch_end=patch_slice_slice
    for i in range(Devide_integer):
        #print(patch_start)
        #print(patch_end)
        ct_volume=image_array[patch_start:patch_end,:,:]
        #print(ct_volume.shape)
        patch_list.append(ct_volume)
        patch_start+=patch_slice_slice
        patch_end+=patch_slice_slice

    last_slice_number_would_be=image_array.shape[0]
    print(last_slice_number_would_be)
    last_patch_When_making_nifty=(patch_slice_slice)-Reminder
    print(last_patch_When_making_nifty)
    Slice_will_start_from_here=last_slice_number_would_be-patch_slice_slice
    print(Slice_will_start_from_here)
    last_patch=image_array[Slice_will_start_from_here:,:,:]
    last_patch.shape
    patch_list.append(last_patch)

    return patch_list,last_patch_When_making_nifty

def Get_Build_model(Input_patch_size,Layer_name,cnn):
    
    '''THIS FUNCTION IS OK EMERALD'''

    #cnn.summary()
    Build_model=tf.keras.models.Model([cnn.inputs], [cnn.get_layer(Layer_name).output, cnn.output])
    #Build_model.summary()
    return Build_model


def Guided_GradCAM_3D(Grad_model,str_ac,str_nac,rest_ac,rest_nac,LABEL):

    # Create a graph that outputs target convolution and output
    # input_ct_io=tf.expand_dims(ct_io, axis=-1)
    # input_ct_io=tf.expand_dims(input_ct_io, axis=0)
    

    the_shape = str_ac.shape
    str_ac = tf.expand_dims(str_ac, axis=0)
    str_nac = tf.expand_dims(str_nac, axis=0)
    rest_ac =  tf.expand_dims(rest_ac, axis=0)
    rest_nac = tf.expand_dims(rest_nac, axis=0)

    
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = Grad_model([str_ac,str_nac,rest_ac,rest_nac])
        loss = predictions[:, int(LABEL)]
    # Extract filters and gradients
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    ##--Guided Gradient Part
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    # Average gradients spatially
    weights = tf.reduce_mean(guided_grads, axis=(0, 1,2))
    # Build a ponderated map of filters according to gradients importance
    cam = np.ones(output.shape[0:3], dtype=np.float32)
    for index, w in enumerate(weights):
        cam += w * output[:, :, :, index]

    capi=resize(cam,(the_shape))
    #print(capi.shape)
    capi = np.maximum(capi,0)
    heatmap = (capi - capi.min()) / (capi.max() - capi.min())
    return heatmap

# generates the grad_cams for a specific layer of the 4-path. Repeat with new layer name to 
# get the rest of the layer outputs
# Input Data: each data must be [15,64,64,1]. LABEL shape = (1,2)



def generate_guided_grad_cam(str_ac,str_nac,rest_ac,rest_nac,LABEL
                             ,Layer_name,cnn,target):
    # Reading the CT
    

    #Class_index=LABEL.argmax(axis=-1)
    #Class_index = Class_index[0,]
    Layer_name=Layer_name

    Input_patch_size=[str_ac.shape[0],str_ac.shape[1],str_ac.shape[2],str_ac.shape[3]]

    Grad_model=Get_Build_model(Input_patch_size,Layer_name,cnn)

    heatmap=Guided_GradCAM_3D(Grad_model,str_ac,str_nac,rest_ac,rest_nac,LABEL)
    
    #heatmap = heatmap[0,:,:,:,:]
    
    # generated the first of the 15 grads
    #cam = cv2.applyColorMap(np.uint8(255*heatmap[0,:,:,:]), cv2.COLORMAP_JET)
    
    
    cam = np.uint8(255*heatmap[0,:,:,:])
    
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    cam = jet_colors[cam[:,:,0]]    
    cam = np.uint8(255*cam)
    
    
    if np.amax(target[0,:,:,:]) <= 1:
        img = target[0,:,:,:]
    else:
        img = target[0,:,:,:]
        TT = np.maximum(img,0)
        img = (TT - TT.min()) / (TT.max() - TT.min())
        #img = target[0,:,:,:] / np.amax(target[0,:,:,:])
    img = img*255

    
    img = np.uint8(np.concatenate((img,)*3, axis=-1))
    
    IMPOSED = cv2.addWeighted(cam, 0.22, img, 1, 0)
    
    IMPOSED = IMPOSED[12:52,12:52,:]
    IMPOSED = cv2.resize(IMPOSED, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    
    plt.imshow(IMPOSED)
    
    
    
    fused = [IMPOSED]
    
    # generate the rest grads of the 15 and stack
    for i in range(1,heatmap.shape[0],1):
        #cam = cv2.applyColorMap(np.uint8(255*heatmap[i,:,:,:]), cv2.COLORMAP_JET)
        
        cam = np.uint8(255*heatmap[i,:,:,:])
        
        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")
    
        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        cam = jet_colors[cam[:,:,0]]         
        cam = np.uint8(255*cam)
        
        if np.amax(target[i,:,:,:])<=1:
            img = target[i,:,:,:]
        else:
            img = target[i,:,:,:]
            TT = np.maximum(img,0)
            img = (TT - TT.min()) / (TT.max() - TT.min())
            #img = target[i,:,:,:] / np.amax(target[i,:,:,:])
        img = img*255
        #cam = np.float32(cam) + np.float32(img)
        
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #img = np.uint8(np.stack((img[:,:,0],)*3, axis=-1))
        
        # https://stackoverflow.com/questions/40119743/convert-a-grayscale-image-to-a-3-channel-image/40119878#40119878
        img = np.uint8(np.concatenate((img,)*3, axis=-1))
        
        IMPOSED = cv2.addWeighted(cam, 0.22, img, 1, 0)
        IMPOSED = IMPOSED[12:52,12:52,:]
        IMPOSED = cv2.resize(IMPOSED, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
        fused.append(IMPOSED)
        #cam = 255 * cam / np.max(cam)
    
    fused = np.stack(fused,axis=0)
    
    # 
    # plt.imshow(img)
    # plt.imshow(fused1)
    
    return fused


# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.

#plot_slices(3, 5, 64, 64, array,save_path,save_name) # array must be [15,64,64,1]


#Layer_name = 'str_ac' # str_ac,str_nac,rest_ac,rest_nac
