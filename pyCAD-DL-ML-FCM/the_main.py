# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 16:20:25 2022

@author: John
"""

'''IMPORT LIBS'''
import os
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, 'C:\\Users\\User\\DSS EXPERIMENTS\\EMERALD_CAD_SCRIPTS\\')
import logging
import numpy as np

'''CONFIGURE LOGS'''



'''CUSTOM SCRIPTS'''
import configurations
import tasks
import routines
import data_preprocessing



'''GET THE CONFIGURATIONS'''
# in dictionary format
import configurations
OPTIONS_PREPROCESSING,OPTIONS_DATA_ANALYTICS,OPTIONS_MODE,OPTIONS_TRAINING,OPTIONS_DICOM_LOAD,OPTIONS_FCM = configurations.get_options ()



#%%
'''FUNCTIONALITIES'''

# LOAD DATA
DATA_REST_AC, DATA_REST_NAC, DATA_STRESS_AC, DATA_STRESS_NAC, LABELS,EXPERT_LABELS,MPI_LABELS, excel_file, ATT, INFO,error,time_seconds = data_preprocessing.load_cad_dicoms_sa001(OPTIONS_DICOM_LOAD)

#%%

'''IMAGE ONLY || CLINICAL ONLY CLASSIFICATION TESTS'''

# TRAIN NETWORK
import tasks
folds_metrics,predictions_all,predictions_all_num,test_labels,duration,history,FINAL_METRICS = tasks.train_image_only (OPTIONS_MODE,OPTIONS_TRAINING,OPTIONS_PREPROCESSING,DATA_REST_AC,DATA_REST_NAC,DATA_STRESS_AC,DATA_STRESS_NAC,ATT,LABELS,EXPERT_LABELS,MPI_LABELS,INFO)


# TRAIN A CLINICAL_ONLY MODEL
import tasks
folds_metrics_clinical,predictions_all_clinical,predictions_all_num_clinical,test_labels_clinical,duration_clinical,history_clinical,FINAL_METRICS_clinical = tasks.train_clinical_only (OPTIONS_MODE,OPTIONS_TRAINING,OPTIONS_PREPROCESSING,DATA_REST_AC,DATA_REST_NAC,DATA_STRESS_AC,DATA_STRESS_NAC,ATT,LABELS,EXPERT_LABELS,INFO)


AGREEMENTS_CLINICAL = routines.metrics_agreement(EXPERT_LABELS,predictions_all_clinical)
#%%

'''DEEP FCM'''


# train an integration
import tasks
dataset,duration,folds_metrics,predictions_all,predictions_all_num,test_labels,duration,FINAL_METRICS, folds_metrics_img,predictions_all_img,predictions_all_num_img,FINAL_METRICS_img,history_cnn,history_ml,cnn = tasks.train_integration (OPTIONS_MODE,OPTIONS_TRAINING,OPTIONS_PREPROCESSING,OPTIONS_FCM,DATA_REST_AC,DATA_REST_NAC,DATA_STRESS_AC,DATA_STRESS_NAC,ATT,LABELS,EXPERT_LABELS,MPI_LABELS,INFO)


AGREEMENTS = routines.metrics_agreement(EXPERT_LABELS,predictions_all)


list_of_vars = [ATT,dataset,EXPERT_LABELS.argmax(axis=-1),FINAL_METRICS,FINAL_METRICS_img,folds_metrics,folds_metrics_img,
                INFO,LABELS,predictions_all,predictions_all_img,predictions_all_num,predictions_all_num_img,
                test_labels,duration,AGREEMENTS]

routines.exporter (list_of_vars, save_path='C:\\Users\\User\\DSS EXPERIMENTS\\EMERALD_CAD_SCRIPTS\\')
#%%

# PLOT RESULTS
routines.history_plots (history_cnn,OPTIONS_DICOM_LOAD['class_names'],predictions_all_num,LABELS)

routines.conf_matrix_plot(FINAL_METRICS['CNF'], OPTIONS_TRAINING['class_names'])

# GRAD CAM


