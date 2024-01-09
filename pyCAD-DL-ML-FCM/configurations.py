# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:20:25 2022

@author: John
"""


def get_options ():
    
    OPTIONS_DICOM_LOAD = {
              'verbose'           : False,
              'example'           : True,
              'normalization'     : False,
              'shuffle'           : True,
              'no_of_slices'      : 15,
              'path'              : "C:\\Users\\User\\EMERALD DATA\\Processed Datasets\\CAD\\1st version (only SA fiels nothing else)\\",
              'label_path'        : "C:\\Users\\User\\EMERALD DATA\\Processed Datasets\\CAD\\TF SPECT_labelsv2_with_normals.xlsx",
              'label_col_name'    : 'pos/neg >70',
              'class_names'       : ["Healthy","Parathyroid"]
              }
    
    
    
    
    
    
    
    OPTIONS_PREPROCESSING = {
                        "W"              : 64,
                        "H"              : 64,
                        "D"              : 15,
                        'shape'          : (15,64,64,1)
        }
    
    # pass to a function which returns: image data in numpy array
    # and auxilliary variables
    
    
    
    OPTIONS_DATA_ANALYTICS = {
                        }
    
    
    # pass to a function and return analytics from the data (e.g. PCA, RF, Correlation, Means etc)
    
    
    
    OPTIONS_MODE = {
                        "cnn"                : "3d_main_70", #3d_main, 3d_main_70, cnn_volume, 3d_main_exp
                        "classifier"         : "rf", #rf, ada, svm, sgd, tree, bag_knn, mlp
                        "grad_cam"           : False,
                        "feature_maps"       : True,
                        "importances"        : True,# only works for RF
                        "grid_search"        : False
        }
    
    
    
    
    
    OPTIONS_TRAINING = {
                        "epochs"             : 80, # set to 75
                        "k-split"            : 10,
                        "validation"         : 10,
                        "classes"            : 2,
                        'tune'               : 'frozen', # 'scratch','number_of_trainable'
                        "class_names"        : ["Healthy", "CAD"],
                        'augmentation'       : False,
                        "batch_size"         : 32,
                        "plot_CM"            : False,
                        "verbose"            : True,
                        "verbose_metrics"    : True,
                        "save_model_after"   : False,
                        "grad-cam-plots"     : False,
                        'dataset-plots'      : False}
    # independent options (global)

    OPTIONS_FCM = {
                        "weight_matrix_path" : "C:\\Users\\User\\DSS EXPERIMENTS\\EMERALD_CAD_SCRIPTS\\",
                        "weight_matrix_name" : "FCM_weights_up.xlsx",
                        "verbose"            : False,
                        'iterations'         : 20,
        }
    
    return OPTIONS_PREPROCESSING,OPTIONS_DATA_ANALYTICS,OPTIONS_MODE,OPTIONS_TRAINING,OPTIONS_DICOM_LOAD,OPTIONS_FCM



