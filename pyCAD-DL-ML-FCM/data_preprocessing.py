# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:21:52 2022

@author: John
"""


import os
import sys
import pandas as pd
from shutil import copy2
import numpy as np
import pydicom as pyd
import matplotlib.pyplot as plt
import datetime
from copy import deepcopy
import time
from keras.utils import to_categorical
import random
import cv2
from scipy.ndimage import zoom



def load_dicom(path, full):
    
    start = time.time()
    
    dicom = pyd.read_file(path)
    img_series = dicom.pixel_array
    
    info = {
    "number_of_imgs":0,
    "machine_model":0,
    "machine_manu":0,
    "slice_thickness":0,
    "study_desc":0,
    "img_max_value":0,
    'slope':0,
    'intercept':0
    }
    
    if full:
        number_of_imgs = int(dicom.NumberOfFrames)
        machine_model = dicom.ManufacturerModelName
        machine_manu = dicom.Manufacturer
        slice_thickness = dicom.SliceThickness
        study_desc = dicom.StudyDescription
        img_max_value = img_series.max()
        
        info = {
            "number_of_imgs":number_of_imgs,
            "machine_model":machine_model,
            "machine_manu":machine_manu,
            "slice_thickness":slice_thickness,
            "study_desc":study_desc,
            "img_max_value":img_max_value,
            'slope':0,
            'intercept':0
            }
    
    return img_series, info
    

def transform_to_HU (scan,info):
    
    ''' 
    
    Input is a numpy array  (slices,width,height). There is no channel here
    because the array is from pydicom load from a CT discom scan
    
    The function brings the array values to the HU scale -1000HU to 1000HU
    
    '''
    
    return (scan * info['slope']) + info['intercept']

def change_window_length (scan):
    
    ''' 
    
    Future EMERALD.
    Input is a numpy array  (slices,width,height). There is no channel here
    because the array is from pydicom load from a CT discom scan.
    The input must be a pure dicom loaded file. No prior HU transformation, else window collapses
    
    The function limits the window length. refer to https://theaisummer.com/medical-image-python/ for more
    Conflicts with transform_to_HU. 
    
    '''
    return scan



def denoise (scan):
    
    ''' 
    
    Future EMERALD.
    Input is a numpy array  (slices,width,height). There is no channel here
    because the array is from pydicom load from a CT discom scan
    
    The input must be firtsly rescaled to HU units.
    
    This is an optional step
    
    '''
    
    return scan




def normalize_from_HU (scan):

    
    
    
    return scan


def normalize_from_pixels (scan):
    
    MIN_BOUND = scan.min()
    MAX_BOUND = scan.max()
    
    scan = (scan - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    scan[scan>1] = 1.
    scan[scan<0] = 0.
    return scan


def att_transform (ATT):
    
    formed = ATT
    formed = formed.astype(float)
    #formed['AGE'] = round(formed['AGE']/120,2)
    #formed['BMI'] = round(formed['BMI']/50,3)
    
    formed['Overweight'] = (formed['BMI'].between(25, 30)).astype(int)
    formed['Obese'] = (formed['BMI'].between(30, 500)).astype(int)
    formed['normal_weight'] = (formed['BMI'] < 30).astype(int)
    
    formed['<40'] = (formed['AGE'] < 40).astype(int)
    formed['40-50'] = (formed['AGE'].between(40, 50)).astype(int)
    formed['50-60'] = (formed['AGE'].between(51, 60)).astype(int)
    formed['>60'] = (formed['AGE'] > 60).astype(int)
    
    formed = formed.drop("BMI",1)
    formed = formed.drop("AGE",1)
    
    #formed['EXPERT DIAGNOSIS: REVERSIBLE DEFECT SIZE (ISCHEMIA)'] = round(formed['EXPERT DIAGNOSIS: REVERSIBLE DEFECT SIZE (ISCHEMIA)']/3,2)
    
    
    return formed

def load_cad_dicoms_sa001(OPTIONS_DICOM_LOAD):
    
    ''' TOTAL LOADER
    
    Give paths to labels and patient folders
    give the column name that is used for reference (1 output mode)
    give the CONFIG
    
    This functions does the following:
        searches in the patient folders and loads the 4 dicoms inside them. The dicoms must have specific names
        loads the dicoms and makes them 4D, channel last format. It keeps only the central slices according to config
        reads the excel file and constructs:
            a. a labels list (binary)
            b. a dataframe with every info
            c. an ATT dataframe which contains only the ML attributes and the patient ID -> this must go to index later
        The ATT file goes for the ML models
        The excel_file contains all the data
        
    For a correct operation, specify the columns that are unecessary in the list excluded_columns. Those columns are excluded from the ATT
    
    All columns must 
    
    
    ''' 
    start = time.time()
    
    path = OPTIONS_DICOM_LOAD['path']
    label_path = OPTIONS_DICOM_LOAD['label_path']
    label_col_name = OPTIONS_DICOM_LOAD['label_col_name']
    
    
    
    ''' EXLUDE AND DISCARD SOME COLUMNS'''
    
    excluded_columns = ['NAME',
                        'EXPERT DIAGNOSIS: STRESS DEFECT SIZE\nNORM=0/SMALL=1/MED=2/LARGE=3',
                        'LM>50','LAD>70','LCX >70','RCA > 70','MULTIVESSEL >70','0,1,2,3 VS',
                        'SCAN 1','SCAN > 1','SCAN 2','pos/neg >70','Polar Map','Expert Diagnosis Binary'
                        ,'EXPERT DIAGNOSIS: REVERSIBLE DEFECT SIZE (ISCHEMIA)']
    
    if OPTIONS_DICOM_LOAD['verbose']:
        print('')
        print('===========================================')
        print('===== INITIALIZING CAD DICOM LOADER  ======')
        print('===========================================')
        print('Datetime: {}'.format(datetime.datetime.now()))
        print('')
        
    # EMPTY DATA LISTS
    DATA_STRESS_AC = []
    DATA_STRESS_NAC = []
    DATA_REST_AC = []
    DATA_REST_NAC = []
    
    # LABELS LIST
    LABELS = []
    EXPERT_LABELS = []
    MPI_LABELS = []
    

    
    
    # LOAD EXCEL FILE
    excel_file = pd.read_excel(label_path)
    
    # VERY IMPORTANT INFORMATION. THE COLUMN OF THE GROUND TRUTH
    GROUND_TRUTH = label_col_name
    column = excel_file.columns.get_loc(GROUND_TRUTH)
    expert_column = excel_file.columns.get_loc('Expert Diagnosis Binary')
    mpi_column = excel_file.columns.get_loc('Polar Map')
    
    # CONSTRUCT THE INFO DATAFRAME   
    columns = ["patient_no",
               "posneg>70",
               "vessels",
               "expert",
               "dicom_path",
               "number_of_imgs",
               "model",
               "manufacturer",
               "thickness",
               "study",
               "max_val"
        ]
    
    INFO = pd.DataFrame(columns = columns)
    
    
    
    
    '''HANDLE THE ATTRIBUTES OF THE FCM MODEL'''
    
    #construct a dataframe with only the attributes
    attributes = deepcopy(excel_file)
    for col in excluded_columns:
        print(col)
        attributes.drop(col,1,inplace=True)
    
    
    # the SEX is in f,m format -> turn to binary
    mapping = {'f': 0, 'm': 1}
    attributes['male'] = attributes['SEX'].map(mapping)
    
    mapping = {'f': 1, 'm': 0}
    attributes['female'] = attributes['SEX'].map(mapping) 
    
    attributes = attributes.drop(['SEX'], axis=1) 

    # ATRIBUTE SET
    ATT = pd.DataFrame(columns = attributes.columns)



    ''' EXLUDE AND DISCARD SOME COLUMNS'''
    
    # FIND THE SUBFOLDERS OF THE MAIN PATH
    folder_patients = os.listdir(path)
    print('FOUND {} PATIENT FOLDERS'.format(len(folder_patients)))
    random.shuffle(folder_patients)
    random.shuffle(folder_patients)
    random.shuffle(folder_patients)
    
    error = []
    # FOR EACH SUBFOLDER (PATEINT CASE)
    for patient in folder_patients:
        if OPTIONS_DICOM_LOAD['verbose']:
            print('')
            print('--> Working on folder: {}'.format(patient))
        the_path = os.path.join(path,patient)
        the_files = os.listdir(the_path)
        
        # EXCEL MATCH
        try:
            pat_id = int(patient[:4])
        except Exception as e:
            print(e)
            continue
        
        try:
            row = excel_file[excel_file['No']==pat_id].index[0]
            
            label = int(excel_file.iloc[row,column])
            label_expert = int(excel_file.iloc[row,expert_column])
            label_mpi = int(excel_file.iloc[row,mpi_column])
            
            
        except Exception as e:
            print (e)
            print ('Patient {} not matched with excel'.format(pat_id))
            continue
    
        
        # MATCH THE FILE NAMES WITH THE EXPECTED REST-STRESS AC-NAC ENTITIES
        ok = True
        for filename in the_files:
            complete_filepath = os.path.join(the_path,filename)
            if "STRESS_AC" in filename:
                stress_ac_path = os.path.join(the_path,filename)
            elif "STRESS_NAC" in filename:
                stress_nac_path = os.path.join(the_path,filename)
            elif "REST_AC" in filename:
                rest_ac_path = os.path.join(the_path,filename)
            elif "REST_NAC" in filename:
                rest_nac_path = os.path.join(the_path,filename)
            else:
                print('Problem in folder. Could not match the stress-rest ac-nac files')
                ok = False
        
        if not ok:
            continue
        else:
            if OPTIONS_DICOM_LOAD['verbose']:
                print('Stress-Rest-AC-NAC files match. Proceeding')
            
            try:    
                # LOAD A STRESS AC
                
                stress_nac,info_stress_nac = load_dicom(path=stress_nac_path, full=False)
                rest_ac,info_rest_ac  = load_dicom(path=rest_ac_path, full=False)
                rest_nac,info_rest_nac  = load_dicom(path=rest_nac_path, full=False)
                stress_ac,info_stress_ac  = load_dicom(path=stress_ac_path, full=True)
                info = info_stress_ac
                
                # print('Patient: {}, Min: {}, Max: {}'.format(pat_id,stress_ac.min(),stress_ac.max()))
                # print('Patient: {}, Min: {}, Max: {}'.format(pat_id,stress_nac.min(),stress_nac.max()))
                # print('Patient: {}, Min: {}, Max: {}'.format(pat_id,rest_ac.min(),rest_ac.max()))
                # print('Patient: {}, Min: {}, Max: {}'.format(pat_id,rest_nac.min(),rest_nac.max()))
                
                #'''TO HU'''
                # there is no need to turn to HU because the CAD dataset is of CT modality
                # it is already in HU. this is because in the DICOM information there is no slope,intercept
                # information. Thus, the image is in the HU scale. However, it is of a specific window
                
                '''NORMALIZATION'''
                if OPTIONS_DICOM_LOAD['normalization']:
                    stress_nac = normalize_from_pixels (stress_nac)
                    rest_ac  = normalize_from_pixels (rest_ac)
                    rest_nac = normalize_from_pixels (rest_nac)
                    stress_ac  = normalize_from_pixels (stress_ac)               
                
                
                '''CROP AND RESIZE OPTIONS'''
                
                stress_nac = stress_nac [:,16:48,16:48] # [:,16:48,16:48]
                stress_nac = zoom(stress_nac, (1, 2, 2))
                
                stress_ac = stress_ac [:,16:48,16:48]
                stress_ac = zoom(stress_ac, (1, 2, 2))
                
                rest_nac = rest_nac [:,16:48,16:48]
                rest_nac = zoom(rest_nac, (1, 2, 2))
                
                rest_ac = rest_ac [:,16:48,16:48]
                rest_ac = zoom(rest_ac, (1, 2, 2))
                
                # import matplotlib.pyplot as plt
                # plt.imshow(stress_nac[10,:,:])
                # plt.imshow(stress_nac2[10,:,:]) 
                
                
                
                ''' REDUCE THE NUMBER OF SLICES '''
                
                central = divmod(stress_ac.shape[0],2)[0]
                requested = OPTIONS_DICOM_LOAD["no_of_slices"]
                left = central - divmod(requested,2)[0]
                right = central + divmod(requested,2)[0] + divmod(requested,2)[1]
                
                # KEEP THE DESIRED NO. OF SLICES

                stress_ac = stress_ac[ left : right,  :,:]
                
                
                # EXPAND DIMS
                stress_ac = np.expand_dims(stress_ac, axis=3)
                
                
                
                
                
                # LOAD A STRESS NAC
                
                # KEEP THE DESIRED NO. OF SLICES
                central = divmod(stress_nac.shape[0],2)[0]
                requested = OPTIONS_DICOM_LOAD["no_of_slices"]
                left = central - divmod(requested,2)[0]
                right = central + divmod(requested,2)[0] + divmod(requested,2)[1]

                stress_nac = stress_nac[  left : right,  :,:]
                
                # EXPAND DIMS
                stress_nac = np.expand_dims(stress_nac, axis=3)            
                
                
                
                
                
                # LOAD A REST AC
                
                # KEEP THE DESIRED NO. OF SLICES
                central = divmod(rest_ac.shape[0],2)[0]
                requested = OPTIONS_DICOM_LOAD["no_of_slices"]
                left = central - divmod(requested,2)[0]
                right = central + divmod(requested,2)[0] + divmod(requested,2)[1]
                rest_ac = rest_ac[  left : right,    :,:]

                # EXPAND DIMS
                rest_ac = np.expand_dims(rest_ac, axis=3)            
                
                
                
                
                
                # LOAD A REST NAC
                
                # KEEP THE DESIRED NO. OF SLICES
                central = divmod(rest_nac.shape[0],2)[0]
                requested = OPTIONS_DICOM_LOAD["no_of_slices"]
                left = central - divmod(requested,2)[0]
                right = central + divmod(requested,2)[0] + divmod(requested,2)[1]
                rest_nac = rest_nac[  left : right,    :,:]

                # EXPAND DIMS
                rest_nac = np.expand_dims(rest_nac, axis=3)
                
                
                DATA_STRESS_AC.append(stress_ac)
                DATA_STRESS_NAC.append(stress_nac)
                DATA_REST_AC.append(rest_ac)
                DATA_REST_NAC.append(rest_nac)
                
    
                LABELS.append(label)
                EXPERT_LABELS.append(label_expert)
                MPI_LABELS.append(label_mpi)
                
                # FIND THE INDEX OF THE ROW WITH PATIENT ID EQUAL TO THE CURRENT ID AND APPEND IT TO THE ATT DATAFRAME
                where_is = attributes.index[attributes['No'] == pat_id].tolist()[0]
                ATT = ATT.append(attributes.iloc[where_is])
                
                INFODICT = {
                    "patient_no":pat_id,
                   "posneg>70":label,
                   "vessels":excel_file.iloc[where_is]['0,1,2,3 VS'],
                   "expert":excel_file.iloc[where_is]['EXPERT DIAGNOSIS: REVERSIBLE DEFECT SIZE (ISCHEMIA)'],
                   "dicom_path":'na',
                   "number_of_imgs":info['number_of_imgs'],
                   "model":info['machine_model'],
                   "manufacturer":info['machine_manu'],
                   "thickness":info['slice_thickness'],
                   "study":info['study_desc'],
                   "max_val":info['img_max_value']
                   }
                
                INFO = INFO.append(INFODICT,ignore_index=True)
                
                assert rest_nac.shape[0]+rest_ac.shape[0]+stress_nac.shape[0]+stress_ac.shape[0] == 4*rest_nac.shape[0], 'img_series do not have the same legnth: {}'.format(pat_id)
                
            except Exception as e:
                print (e)
                error.append(pat_id)
    
    DATA_REST_AC = np.stack(DATA_REST_AC)
    DATA_REST_NAC = np.stack(DATA_REST_NAC)
    DATA_STRESS_AC = np.stack(DATA_STRESS_AC)
    DATA_STRESS_NAC = np.stack(DATA_STRESS_NAC)
    
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # NEED TO RETURN ICA_LABELS, HUMAN_LABELS
    # NEED TO RETURN ATT ARRAY WITHOUT ANY LABELS AND WITHOUT PATIENT NO
    
    ATT = ATT.drop("No",1)
    ATT = att_transform (ATT)
    
    
    '''FUNCTION TO TURN ATT TO NUMERIC NORMALIZED'''
    
    
    
    LABELS = to_categorical(LABELS)
    EXPERT_LABELS = to_categorical(EXPERT_LABELS)
    MPI_LABELS = to_categorical(MPI_LABELS)
    
    
    # Sample image
    if OPTIONS_DICOM_LOAD['example']:
        for i in range (DATA_REST_AC.shape[1]):
            img = DATA_REST_AC[1,8,:,:,:]
            import matplotlib.pyplot as plt
            plt.imshow(img)
    
    
    end = time.time()
    
    time_seconds = round(end-start,3)
    print ("Time taken: {} seconds".format (time_seconds))
    return DATA_REST_AC, DATA_REST_NAC, DATA_STRESS_AC, DATA_STRESS_NAC, LABELS,EXPERT_LABELS,MPI_LABELS, excel_file, ATT, INFO,error,time_seconds