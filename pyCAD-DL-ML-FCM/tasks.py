# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:07:24 2022

@author: John
"""
import routines
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
import datetime
import os
import tensorflow as tf
# start = time.time()
# end = time.time()
# duration = (end - start)

import data_preprocessing
from image_data_gen import VoxelDataGenerator
from volume_generator import emerald_4d_augmentation
import grad_cam_3d
import fcm_functions
import image_3d_generator_adaption

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_image_only (OPTIONS_MODE,OPTIONS_TRAINING,OPTIONS_PREPROCESSING,DATA_REST_AC,DATA_REST_NAC,DATA_STRESS_AC,DATA_STRESS_NAC,ATT,LABELS,EXPERT_LABELS,MPI_LABELS,INFO):
    start = time.time()
    
    # storage
    predictions_all = np.empty(0) # here, every fold predictions will be kept
    test_labels = np.empty([0,OPTIONS_TRAINING['classes']])
    test_labels_onehot = np.empty(0)#here, every fold labels are kept
    predictions_all_num = np.empty([0,OPTIONS_TRAINING['classes']])
    folds_metrics = []
    
    summ = True
    
    for train_index, test_index in KFold(OPTIONS_TRAINING['k-split']).split(LABELS):

            
        
        # KFOLD SPLIT DATA
        train_stress_ac, test_stress_ac = DATA_STRESS_AC[train_index], DATA_STRESS_AC[test_index]
        train_stress_nac, test_stress_nac = DATA_STRESS_NAC[train_index], DATA_STRESS_NAC[test_index]
        train_rest_ac, test_rest_ac = DATA_REST_AC[train_index], DATA_REST_AC[test_index]
        train_rest_nac, test_rest_nac = DATA_REST_NAC[train_index], DATA_REST_NAC[test_index]
        
        # KFOLD SPLIT LABELS
        train_label_ica, test_label_ica = LABELS[train_index], LABELS[test_index]
        train_label_expert, test_label_expert = EXPERT_LABELS[train_index], EXPERT_LABELS[test_index]
        
        # assign ground truth
        train_label = train_label_ica
        test_label = test_label_ica
        
        # PICK THE MODEL: CALL ANOTHER FUNCTION TO PICK IT
        model = routines.pick_model(OPTIONS_MODE, OPTIONS_PREPROCESSING, OPTIONS_TRAINING)
        if summ:
            summ = False
            model.summary()
        
        # params
        epochs = int(OPTIONS_TRAINING['epochs'])
        batch_size = int(OPTIONS_TRAINING['batch_size'])
        

        
        
        # PLACEHOLDER
        if OPTIONS_TRAINING['augmentation']:
            print ('Under Development')
            augmentation = True

        else:
            augmentation = False
            
        
        # TRAIN
        
        if augmentation:
            print ('AUGMENTING....')
            aug_rest_ac,aug_rest_nac,aug_stress_ac,aug_stress_nac,train_label = emerald_4d_augmentation (train_stress_ac,train_stress_nac,train_rest_ac,train_rest_nac,train_label,OPTIONS_PREPROCESSING,nofaug = 2)
            history = model.fit([aug_stress_ac,aug_stress_nac,aug_rest_ac,aug_rest_nac], train_label, validation_split=0.10, epochs=epochs, batch_size=batch_size)
            
        else:

            if OPTIONS_MODE['cnn'] == 'cnn_volume':
                train_dataset = np.concatenate([train_stress_ac,train_stress_nac,train_rest_ac,train_rest_nac],axis=1)
                test_dataset = np.concatenate([test_stress_ac,test_stress_nac,test_rest_ac,test_rest_nac],axis=1)
                history = model.fit(train_dataset, train_label, validation_split=0.10, epochs=epochs, batch_size=batch_size)
            else:
                history = model.fit([train_stress_ac,train_stress_nac,train_rest_ac,train_rest_nac], train_label, validation_split=0.10, epochs=epochs, batch_size=batch_size)
                            
        # predict the unseen
        if OPTIONS_MODE['cnn'] == 'cnn_volume':
            predict = model.predict(test_dataset)
        else:
            predict = model.predict([test_stress_ac,test_stress_nac,test_rest_ac,test_rest_nac])
        predict_num = predict
        predict = predict.argmax(axis=-1)
        
        
        test_label_onehot = np.argmax(test_label, axis=-1) #make the labels 1column array
        predictions_all = np.concatenate([predictions_all, predict])
        predictions_all_num = np.concatenate([predictions_all_num, predict_num])
        test_labels = np.concatenate([test_labels, test_label])
        test_labels_onehot = np.concatenate([test_labels_onehot, test_label_onehot])
        
        # CALL METRICS FUNCTION
        fold_metrics = routines.metrics(predict, predict_num, test_label, test_label_onehot)
        print ('Test Accuracy: {}, SEN: {}, SPE {}'.format(round(fold_metrics['Accuracy'],2),round(fold_metrics['Sensitivity'],2),round(fold_metrics['Specificity'],2)))
        folds_metrics.append(fold_metrics)
        
    FINAL_METRICS = routines.metrics(predictions_all, predictions_all_num, test_labels, test_labels_onehot)     
    print('')
    print ('FINAL Test Accuracy: {}, SEN: {}, SPE {}'.format(round(FINAL_METRICS['Accuracy'],2),round(FINAL_METRICS['Sensitivity'],2),round(FINAL_METRICS['Specificity'],2)))
    
    end = time.time()
    duration = round(end - start,2) 
    return folds_metrics,predictions_all,predictions_all_num,test_labels,duration,history,FINAL_METRICS

    

def train_clinical_only (OPTIONS_MODE,OPTIONS_TRAINING,OPTIONS_PREPROCESSING,DATA_REST_AC,DATA_REST_NAC,DATA_STRESS_AC,DATA_STRESS_NAC,ATT,LABELS,EXPERT_LABELS,INFO):
    start = time.time()
    from keras.utils.np_utils import to_categorical 
    
    # storage
    predictions_all = np.empty(0) # here, every fold predictions will be kept
    test_labels = np.empty([0,OPTIONS_TRAINING['classes']])
    test_labels_onehot = np.empty(0)#here, every fold labels are kept
    predictions_all_num = np.empty([0,OPTIONS_TRAINING['classes']])
    folds_metrics = []
    
    summ = True
    
    dataset = np.array(ATT.fillna(0))
    
    for train_index, test_index in KFold(OPTIONS_TRAINING['k-split']).split(LABELS):

        train_clinical, test_clinical = dataset[train_index], dataset[test_index]
        train_clinical = train_clinical.astype(float)
             
        # KFOLD SPLIT LABELS
        train_label_ica, test_label_ica = LABELS[train_index], LABELS[test_index]
        train_label_expert, test_label_expert = EXPERT_LABELS[train_index], EXPERT_LABELS[test_index]
        
        # assign ground truth
        train_label = train_label_ica
        test_label = test_label_ica
        
        # PICK THE MODEL: CALL ANOTHER FUNCTION TO PICK IT
        model = routines.pick_model_clinical(OPTIONS_MODE, OPTIONS_PREPROCESSING, OPTIONS_TRAINING)


        # PLACEHOLDER
        if OPTIONS_TRAINING['augmentation']:
            print ('Under Development')
            augmentation = False

        else:
            augmentation = False
            
        
        # TRAIN
        
        tr = np.argmax(train_label, axis=-1)
        model.fit(train_clinical,tr)
        #f_i = model.feature_importances_
        if OPTIONS_MODE['grid_search']:
            print('Best params: {}'.format(model.best_params_))
            OPTIONS_MODE['grid_search'] = False
    
        # predict the unseen
        predict = model.predict(test_clinical)
        predict_num = predict
        #predict = predict.argmax(axis=-1)
        
        
        test_label_onehot = np.argmax(test_label, axis=-1) #make the labels 1column array
        predictions_all = np.concatenate([predictions_all, predict])
        predictions_all_num = np.concatenate([predictions_all_num, to_categorical(predict_num, num_classes=2)])
        test_labels = np.concatenate([test_labels, test_label])
        test_labels_onehot = np.concatenate([test_labels_onehot, test_label_onehot])
        
        # CALL METRICS FUNCTION
        fold_metrics = routines.metrics(predict, to_categorical(predict_num, num_classes=2), test_label, test_label_onehot)
        print ('Test Accuracy: {}, SEN: {}, SPE {}'.format(round(fold_metrics['Accuracy'],2),round(fold_metrics['Sensitivity'],2),round(fold_metrics['Specificity'],2)))
        folds_metrics.append(fold_metrics)
    
    importance_dict = {}
    if OPTIONS_MODE['classifier'] == 'rf':
        try:
            model.fit(dataset, LABELS)
            f_i = model.feature_importances_
            for col,imp in zip(ATT.columns,f_i):
                importance_dict.update({col: round(imp,2)})
        except Exception as e:
            print (e)
    else: importance_dict = {}
        
            
    
    FINAL_METRICS = routines.metrics(predictions_all, predictions_all_num, test_labels, test_labels_onehot)
    print('')
    print ('FINAL Test Accuracy: {}, SEN: {}, SPE {}'.format(round(FINAL_METRICS['Accuracy'],2),round(FINAL_METRICS['Sensitivity'],2),round(FINAL_METRICS['Specificity'],2)))
    
    history = [model,importance_dict]
    end = time.time()
    duration = round(end - start,2) 
    return folds_metrics,predictions_all,predictions_all_num,test_labels,duration,history,FINAL_METRICS





'''


RANDOM FOREST + CNN INTEGRATION



'''

def train_integration (OPTIONS_MODE,OPTIONS_TRAINING,OPTIONS_PREPROCESSING,OPTIONS_FCM,DATA_REST_AC,DATA_REST_NAC,DATA_STRESS_AC,DATA_STRESS_NAC,ATT,LABELS,EXPERT_LABELS,MPI_LABELS,INFO):
    start = time.time()
    
    # storage
    predictions_all_img = np.empty(0) # here, every fold predictions will be kept
    test_labels_img = np.empty([0,OPTIONS_TRAINING['classes']])
    test_labels_onehot_img = np.empty(0)#here, every fold labels are kept
    predictions_all_num_img = np.empty([0,OPTIONS_TRAINING['classes']])
    folds_metrics_img = []


    # storage
    predictions_all = np.empty(0) # here, every fold predictions will be kept
    test_labels = np.empty([0,OPTIONS_TRAINING['classes']])
    test_labels_onehot = np.empty(0)#here, every fold labels are kept
    predictions_all_num = np.empty([0,OPTIONS_TRAINING['classes']])
    folds_metrics = []

  
    dataset = np.array(ATT.fillna(0))
    summ = True

    now = datetime.datetime.now()
    now = datetime.datetime.strftime(now, '%Y-%m-%d-%H-%M-%S')
    
    for train_index, test_index in KFold(OPTIONS_TRAINING['k-split']).split(LABELS):

            
        
        # KFOLD SPLIT DATA
        train_stress_ac, test_stress_ac = DATA_STRESS_AC[train_index], DATA_STRESS_AC[test_index]
        train_stress_nac, test_stress_nac = DATA_STRESS_NAC[train_index], DATA_STRESS_NAC[test_index]
        train_rest_ac, test_rest_ac = DATA_REST_AC[train_index], DATA_REST_AC[test_index]
        train_rest_nac, test_rest_nac = DATA_REST_NAC[train_index], DATA_REST_NAC[test_index]
        
        
        train_clinical, test_clinical = dataset[train_index], dataset[test_index]
        train_clinical = train_clinical.astype(float)        
        
        
        # KFOLD SPLIT LABELS
        train_label_ica, test_label_ica = LABELS[train_index], LABELS[test_index]
        train_label_expert, test_label_expert = EXPERT_LABELS[train_index], EXPERT_LABELS[test_index]
        train_label_mpi, test_label_mpi = MPI_LABELS[train_index], MPI_LABELS[test_index]
        
        # assign ground truth
        train_label = train_label_expert
        test_label = test_label_expert
        
        #assign info
        info_fold_train = np.array(INFO)[train_index]
        info_fold_test = np.array(INFO)[test_index]
        
        # PICK THE MODEL: CALL ANOTHER FUNCTION TO PICK IT
        model = routines.pick_model(OPTIONS_MODE, OPTIONS_PREPROCESSING, OPTIONS_TRAINING)
        if summ:
            model.summary()
            summ = False
        
        # params
        epochs = int(OPTIONS_TRAINING['epochs'])
        batch_size = int(OPTIONS_TRAINING['batch_size'])
        
        
        # PLACEHOLDER
        if OPTIONS_TRAINING['augmentation']:
            print ('Under Development')
            augmentation = True

        else:
            augmentation = False
            
        
        # TRAIN
        
        if augmentation:
            # if OPTIONS_MODE['cnn'] == 'cnn_volume':
            #     train_dataset = np.concatenate([train_stress_ac,train_stress_nac,train_rest_ac,train_rest_nac],axis=1)
            #     test_dataset = np.concatenate([test_stress_ac,test_stress_nac,test_rest_ac,test_rest_nac],axis=1)
                
            # import image_3d_generator_adaption
            # l= image_3d_generator_adaption.ImageDataGenerator(rotation_range = 20)
            # history_cnn = model.fit(l.flow(train_dataset,train_label,batch_size=batch_size), steps_per_epoch=len(train_stress_ac) // batch_size, epochs=epochs)
            print ('AUGMENTING....')
            aug_rest_ac,aug_rest_nac,aug_stress_ac,aug_stress_nac,train_label = emerald_4d_augmentation (train_stress_ac,train_stress_nac,train_rest_ac,train_rest_nac,train_label,OPTIONS_PREPROCESSING,nofaug = 2)
            history = model.fit([aug_stress_ac,aug_stress_nac,aug_rest_ac,aug_rest_nac], train_label, epochs=epochs, batch_size=batch_size)
                        
        else:
            
            # with validation split
            #history_cnn = model.fit([train_stress_ac,train_stress_nac,train_rest_ac,train_rest_nac], train_label, validation_split=0.10, epochs=epochs, batch_size=batch_size)
            if OPTIONS_MODE['cnn'] == 'cnn_volume':
                train_dataset = np.concatenate([train_stress_ac,train_stress_nac,train_rest_ac,train_rest_nac],axis=1)
                test_dataset = np.concatenate([test_stress_ac,test_stress_nac,test_rest_ac,test_rest_nac],axis=1)
                history_cnn = model.fit(train_dataset, train_label, epochs=epochs, batch_size=batch_size)

            else:
            # without validation split
            
                callback = tf.keras.callbacks.EarlyStopping(
                                                    monitor='val_accuracy',
                                                    min_delta=0.01,
                                                    patience=30,
                                                    verbose=1,
                                                    mode='auto',
                                                    baseline=0.85,
                                                    restore_best_weights=True
                                                )
            
            
            
            
            
                history_cnn = model.fit([train_stress_ac,train_stress_nac,train_rest_ac,train_rest_nac], train_label, validation_split = 0.1, epochs=epochs, batch_size=batch_size,callbacks=[callback])
        

    
        # predict the unseen
        if OPTIONS_MODE['cnn'] == 'cnn_volume':
            predict_img = model.predict(test_dataset)
        else:
            predict_img = model.predict([test_stress_ac,test_stress_nac,test_rest_ac,test_rest_nac])
        predict_num_img = predict_img
        predict_img = predict_img.argmax(axis=-1)
        
        
        test_label_onehot_img = np.argmax(test_label, axis=-1) #make the labels 1column array
        predictions_all_img = np.concatenate([predictions_all_img, predict_img])
        predictions_all_num_img = np.concatenate([predictions_all_num_img, predict_num_img])
        test_labels_img = np.concatenate([test_labels_img, test_label])
        test_labels_onehot_img = np.concatenate([test_labels_onehot_img, test_label_onehot_img])
        
        # CALL METRICS FUNCTION
        fold_metrics_img = routines.metrics(predict_img, predict_num_img, test_label, test_label_onehot_img)
        print ('IMG-> Test Accuracy: {}, SEN: {}, SPE {}'.format(round(fold_metrics_img['Accuracy'],2),round(fold_metrics_img['Sensitivity'],2),round(fold_metrics_img['Specificity'],2)))
        folds_metrics_img.append(fold_metrics_img)
        
        
        
        '''DATA PLOTS'''

        if OPTIONS_TRAINING['dataset-plots']:
            
            save_path = 'C:\\Users\\User\\DSS EXPERIMENTS\\EMERALD_CAD_RESULTS\\Dataset Plots\\'
            new_folder = os.path.join(save_path,now)
            if not os.path.exists(new_folder):
                os.mkdir(new_folder)
            for i in range(len(info_fold_test)):
                pat_id = info_fold_test [i,0]
                
                label = test_label_expert [i,1]
                if label==1: 
                    label = 'RefCAD' 
                else: 
                    label='RefHEAL'
                
                pred = predict_img [i]
                
                if pred==1: 
                    pred = 'PredCAD' 
                else: 
                    pred='PredHEAL'
                    
                name = 'pat_{}_{}_{}'.format(pat_id,label,pred)
                grad_name = 'pat_{}_{}_{}_GRAD_'.format(pat_id,label,pred)
                
                # PLOT THE INITIAL SLICES
                grad_cam_3d.plot_slices(3, 5, 64, 64, test_stress_ac[i,:,:,:,:],save_path=new_folder,save_name=name+'stressAC',grad=False)
                grad_cam_3d.plot_slices(3, 5, 64, 64, test_stress_nac[i,:,:,:,:],save_path=new_folder,save_name=name+'stressNAC',grad=False)
                grad_cam_3d.plot_slices(3, 5, 64, 64, test_rest_ac[i,:,:,:,:],save_path=new_folder,save_name=name+'restAC',grad=False)
                grad_cam_3d.plot_slices(3, 5, 64, 64, test_rest_nac[i,:,:,:,:],save_path=new_folder,save_name=name+'restNAC',grad=False)
            
            
            
            
        ''' 
        
        APLY GRAD CAM
        
        
        '''
        
        if OPTIONS_TRAINING['grad-cam-plots']:
            save_path_grads = 'C:\\Users\\User\\DSS EXPERIMENTS\\EMERALD_CAD_RESULTS\\Grad CAMs\\'
            new_grad_folder = os.path.join(save_path_grads,now)
            
            if not os.path.exists(new_grad_folder):
                os.mkdir(new_grad_folder)
                
                
            for i in range(len(info_fold_test)):
                pat_id = info_fold_test [i,0]
                
                label = test_label_expert [i,1]
                if label==1: 
                    label = 'RefCAD' 
                else: 
                    label='RefHEAL'
                
                pred = predict_img [i]
                
                if pred==1: 
                    pred = 'PredCAD' 
                else: 
                    pred='PredHEAL'
                    
                grad_name = 'pat_{}_{}_{}_GRAD_'.format(pat_id,label,pred)
            
                the_label = test_label_expert[i,1]
                
                
                # GET THE FUSED GRAD CAM IMAGE FOR STRESS AC
                fused_str_ac = grad_cam_3d.generate_guided_grad_cam(test_stress_ac[i,:,:,:,:],test_stress_nac[i,:,:,:,:],
                                                                    test_rest_ac[i,:,:,:,:],test_rest_nac[i,:,:,:,:],LABEL=the_label,
                                             Layer_name='str_ac',cnn=model,target=test_stress_ac[i,:,:,:,:])
                
                
                # PLOT THE FUSED SLICES
                grad_cam_3d.plot_slices_grad(3, 5, 64, 64, fused_str_ac,save_path=new_grad_folder,save_name=grad_name+'stressAC')
                
                # sample = fused_str_ac[10,:,:,:]
                # import matplotlib.pyplot as plt
                # plt.imshow(sample)     
                
                # GET THE FUSED GRAD CAM IMAGE FOR STRESS NAC
                fused_str_nac = grad_cam_3d.generate_guided_grad_cam(test_stress_ac[i,:,:,:,:],test_stress_nac[i,:,:,:,:],
                                                                    test_rest_ac[i,:,:,:,:],test_rest_nac[i,:,:,:,:],the_label,
                                              Layer_name='str_nac',cnn=model,target=test_stress_nac[i,:,:,:,:])
                # PLOT THE FUSED SLICES
                grad_cam_3d.plot_slices(3, 5, 64, 64, fused_str_nac,save_path=new_grad_folder,save_name=grad_name+'stressNAC')
                
                
                # GET THE FUSED GRAD CAM IMAGE FOR REST AC
                fused_rest_ac = grad_cam_3d.generate_guided_grad_cam(test_stress_ac[i,:,:,:,:],test_stress_nac[i,:,:,:,:],
                                                                    test_rest_ac[i,:,:,:,:],test_rest_nac[i,:,:,:,:],the_label,
                                              Layer_name='rest_ac',cnn=model,target=test_rest_ac[i,:,:,:,:])
                # PLOT THE FUSED SLICES
                grad_cam_3d.plot_slices(3, 5, 64, 64, fused_rest_ac,save_path=new_grad_folder,save_name=grad_name+'restAC')
                
                # GET THE FUSED GRAD CAM IMAGE FOR REST NAC
                fused_rest_nac = grad_cam_3d.generate_guided_grad_cam(test_stress_ac[i,:,:,:,:],test_stress_nac[i,:,:,:,:],
                                                                    test_rest_ac[i,:,:,:,:],test_rest_nac[i,:,:,:,:],the_label,
                                              Layer_name='rest_nac',cnn=model,target=test_rest_nac[i,:,:,:,:])
                # PLOT THE FUSED SLICES
                grad_cam_3d.plot_slices(3, 5, 64, 64, fused_rest_nac,save_path=new_grad_folder,save_name=grad_name+'restNAC')
        
    
        
        
    ##################################################
    # END OF FOLDS FOR IMAGE DATA
    #######################################################



    
    '''INTEGRATION'''

    ##################################################
    # START OF 10-FOLD FOR CLINICAL DATA PLUS THE PREDICTION OF THE CNN
    #######################################################
    
    if OPTIONS_MODE['classifier'] != 'FCM':
        pr = tf.keras.utils.to_categorical(predictions_all_img)
        dataset = np.concatenate((dataset,pr),axis=1)
        #dataset = dataset.astype(str)
    else:
        # insert two extra columns to the dataset. Those columns contain the prediciton of the CNN in
        # ctegorical format one-hot-encoded
        pr = tf.keras.utils.to_categorical(predictions_all_img)
        dataset = np.concatenate((dataset,pr),axis=1)
        
        # insert the "FCM-prediction column", because this is the output concept of the FCM
        # this column is filled with zeros (initial state of the FCM is zeroed)
        dataset = np.concatenate((dataset,np.zeros((len(dataset),1))),axis=1)
        
    if OPTIONS_MODE['classifier'] != 'FCM':
        for train_index, test_index in KFold(OPTIONS_TRAINING['k-split']).split(LABELS):
    
            train_clinical, test_clinical = dataset[train_index], dataset[test_index]
            train_clinical = train_clinical.astype(float)
                 
            # KFOLD SPLIT LABELS
            train_label_ica, test_label_ica = LABELS[train_index], LABELS[test_index]
            train_label_expert, test_label_expert = EXPERT_LABELS[train_index], EXPERT_LABELS[test_index]
            
            # assign ground truth
            train_label = train_label_ica
            test_label = test_label_ica
        
            model_ml = routines.pick_model_clinical(OPTIONS_MODE, OPTIONS_PREPROCESSING, OPTIONS_TRAINING)        
            model_ml.fit(train_clinical, train_label)
            
            
            
            # predict the unseen
            predict = model_ml.predict(test_clinical)
            predict_num = predict
            predict = predict.argmax(axis=-1)
            
            
            test_label_onehot = np.argmax(test_label, axis=-1) #make the labels 1column array
            predictions_all = np.concatenate([predictions_all, predict])
            predictions_all_num = np.concatenate([predictions_all_num, predict_num])
            test_labels = np.concatenate([test_labels, test_label])
            test_labels_onehot = np.concatenate([test_labels_onehot, test_label_onehot])
            
            # CALL METRICS FUNCTION
            fold_metrics = routines.metrics(predict, predict_num, test_label, test_label_onehot)
            print ('Test Accuracy: {}, SEN: {}, SPE {}'.format(round(fold_metrics['Accuracy'],2),round(fold_metrics['Sensitivity'],2),round(fold_metrics['Specificity'],2)))
            folds_metrics.append(fold_metrics)
    else:
        
        
        binary_labels,predictions_fcm_num,predictions_fcm_num_2,threshold_of_interest,FCM_model,weight_matrix_np = fcm_functions.emerald_fcm_packet (dataset,LABELS,OPTIONS_FCM)
        
        predictions_all = binary_labels
        predictions_all_num = predictions_fcm_num_2
        test_labels = LABELS
        test_labels_onehot = np.argmax(LABELS,axis=-1)

        folds_metrics = [0]
    
    
    
    if OPTIONS_MODE['classifier'] == 'rf':
        model_ml.fit(dataset, LABELS)
        f_i = model_ml.feature_importances_
        importance_dict = {}
        columns = list(ATT.columns)
        columns.append('CNN_Healthy')
        columns.append('CNN_CAD')
        for col,imp in zip(columns,f_i):
            importance_dict.update({col: round(imp,3)})
    elif OPTIONS_MODE['classifier'] == 'FCM':
        importance_dict = weight_matrix_np
        model_ml = FCM_model
    else: importance_dict = 0        



        
        
    FINAL_METRICS_img = routines.metrics(predictions_all_img, predictions_all_num_img, test_labels_img, test_labels_onehot_img)     

    
    FINAL_METRICS = routines.metrics(predictions_all, predictions_all_num, test_labels, test_labels_onehot)    
    print('')
    print ('FINAL Test Accuracy: {}, SEN: {}, SPE {}'.format(round(FINAL_METRICS['Accuracy'],2),round(FINAL_METRICS['Sensitivity'],2),round(FINAL_METRICS['Specificity'],2)))
    
    if OPTIONS_MODE['classifier'] != 'FCM':
        history_ml = [model_ml,importance_dict]
    else:
        history_ml = [model_ml,importance_dict,threshold_of_interest]
    
    
    
    end = time.time()
    duration = round(end - start,2) 


    return dataset,duration,folds_metrics,predictions_all,predictions_all_num,test_labels,duration,FINAL_METRICS, folds_metrics_img,predictions_all_img,predictions_all_num_img,FINAL_METRICS_img,history_cnn,history_ml,model






'''APP DEVELOPMENT FUNCTIONS: FUTURE'''



def load_trained_model ():
    return

def predict_new_sinle_input ():
    return

