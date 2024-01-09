##################################################################################################################
##################################################################################################################
'''




    EMERALD: PYTHON IMPLEMENTATION OF FUZZY COGNITIVE MAPS
    
    based on: https://github.com/mpuheim/PyOpenFCM





'''
##################################################################################################################
##################################################################################################################



import sys
sys.path.append('C:\\Users\\User\\DSS EXPERIMENTS\\EMERALD_CAD_SCRIPTS\\PyOpenFCM-master\\')
from fcmlib import FCM
import numpy as np
import os
import pandas as pd
from sklearn import metrics


def fcm_construct (the_instance, weight_matrix_np, verbose):
    
    '''
    
    Constructs the FCM model based on the arguments
    
    instant: a numpy array of size [1,attributes_number]. The last att must be the output
    weight_matrix: a numpy array of size [attributes_number,attributes_number]
    
    '''
    if verbose:
        print ('You gave me {} attributes'.format(the_instance.shape[0]))
        print ('The last attribute in the output concept, which is initiated at 0 from the tasks.py process')
        print ('The attributes {} and {}, the two attributes before the last, are the CNN predictions'.format(the_instance.shape[0]-1,the_instance.shape[0]-2))
        
    
    # initiate the FCM
    FCM_model = FCM(C1 = the_instance[0,0]) # map=FCM(C1=0.6,C2=0.4
    
    # initiate the rest of the concepts
    for i in range(2,the_instance.shape[0]+1,1):
        FCM_model["C{}".format(i)] = the_instance[i-1,0]
    
    if verbose:
        print('This is the constructed FCM')
        print(FCM_model)
    
    output_concept_name = 'C{}'.format(the_instance.shape[0]+1)

    # connect all concepts with the last
    for i in range(1,the_instance.shape[0],1):
        FCM_model.connect("C{}".format(i),"C{}".format(the_instance.shape[0]))
        if verbose:
            print("C{}".format(i),"C{}".format(the_instance.shape[0]))
    
    
    # Set the relations of allo concepts with the output
    for i in range(1,the_instance.shape[0]): # from 1 to 25
        affected = 'C{}'.format(the_instance.shape[0])
        affected_by = 'C{}'.format(i)
        weight = weight_matrix_np[i-1,the_instance.shape[0]-1]
        if verbose:
            print('Setting relation of {}->{} with {} weight'.format(affected_by,affected,weight))
        
        FCM_model[affected].relation.get()
        FCM_model[affected].relation.set(affected_by,weight)
        
    
    ############################################################################
    #
    #
    #  INTER-CONCEPT RELATIONS
    #
    #
    ###########################################################################
    
    for i in range (weight_matrix_np.shape[0]):
        for j in range (weight_matrix_np.shape[0]):
            if i==j or i==weight_matrix_np.shape[0]-2 or j==weight_matrix_np.shape[0]-1:
                continue
            if weight_matrix_np[i,j] != 0:
                FCM_model.connect("C{}".format(i+1),"C{}".format(j+1))
                affected = 'C{}'.format(j+1)
                affected_by = 'C{}'.format(i+1)
                weight = weight_matrix_np[i,j]
                if verbose:
                    print('Setting relation of {}->{} with {} weight'.format(affected_by,affected,weight))
                FCM_model[affected].relation.get()
                FCM_model[affected].relation.set(affected_by,weight)
    
    return FCM_model


def fcm_run (FCM_model, iterations,verbose):
    
    '''
    
    Runs the FCM model
    
    iterations: the number of updates
    
    '''
    for i in range (iterations):
        FCM_model.update()
        if verbose:
            print(FCM_model['C{}'.format(len(FCM_model))])
    
    prediction = FCM_model['C{}'.format(len(FCM_model))].value
    
    return FCM_model,prediction

def fcm_save_json (FCM_model,path):
    
    '''
    
    Exports the FCM to json format
    
    
    '''
    
    return


def emerald_fcm_decision_maker (OPTIONS_FCM, predictions_fcm_num,labels):
    
    
    '''
    
    1. receives labels and predictions
    2. defines the threshold
    3. returns: binary predictions, threshold, fuzzy predictions

    
    
    '''
    
    binary_labels = []
    
    fpr, tpr, thresholds = metrics.roc_curve(np.argmax(labels,axis=-1), predictions_fcm_num)
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
    # the first threshold of the array is nonsense
    thresholds = np.delete(thresholds,0)
    
    
    threshold_of_interest = thresholds[np.argmax(tpr - fpr)]
    
    for i in range (len(predictions_fcm_num)):
        if predictions_fcm_num[i]>threshold_of_interest:
            binary_labels.append(1)
        else:
            binary_labels.append(0)
            
    
    
    return binary_labels,threshold_of_interest


def emerald_fcm_packet (dataset,LABELS,OPTIONS_FCM):
    
    '''
    
    FCM MAIN
    
    1. receives a numpy array of [number_of_instants,number of attributes]
    2. for each instant, it calls the emerald_fcm function to classify it
    3. merges the predictions
    4. Defines the threshold
    4. returns trained model,predictions
    
    !! This is not suitable for one instance classification, because the threshold is
    decided by the data packet - a new function is needed
    
    '''
    
    
    print ('')
    print('===============================================================================')
    print ('')
    
    print ('EMERALD IMPLEMENTATION OF FUZZY COGNITIVE MAPS')
    
    print ('')
    print('===============================================================================')
    print ('')   
    
    # load the weights
    path = os.path.join(OPTIONS_FCM['weight_matrix_path'],OPTIONS_FCM['weight_matrix_name'])
    weight_matrix_pd = pd.read_excel(path)
    weight_matrix_pd.index = weight_matrix_pd[weight_matrix_pd.columns[0]]
    weight_matrix_pd = weight_matrix_pd.drop(weight_matrix_pd.columns[0],axis=1)
    weight_matrix_np = np.array(weight_matrix_pd)
    
    
    predictions_fcm_num = []
    
    # for each intance of the dataset (for each patient case practically)
    for i in range(len(dataset)):
        
        # get the instance
        the_instance = dataset[i,:].reshape(-1,1)
        
        # constuct the FCM
        FCM_model = fcm_construct (the_instance, weight_matrix_np, verbose=OPTIONS_FCM['verbose'])
        
        # run the FCM
        FCM_model,prediction = fcm_run (FCM_model, iterations=OPTIONS_FCM['iterations'],verbose=OPTIONS_FCM['verbose'])
        
        # merge predictions
        predictions_fcm_num.append(prediction)
        
    
    # np.array the list of predicted values
    predictions_fcm_num = np.array(predictions_fcm_num)
    reverse_predictions = 1 - predictions_fcm_num
    
    predictions_fcm_num_2 = np.stack((reverse_predictions,predictions_fcm_num),axis=1)
    
    
    # define the threshold
    binary_labels,threshold_of_interest = emerald_fcm_decision_maker (OPTIONS_FCM, predictions_fcm_num,LABELS)
    binary_labels = np.array(binary_labels)   
        
    
    return binary_labels,predictions_fcm_num,predictions_fcm_num_2,threshold_of_interest,FCM_model,weight_matrix_np

def emerald_fcm_predict_one (dataset,OPTIONS_FCM):
    
    '''
    
    This is a function to classify just one instance
    1. Build the MAP
    2. Runs is
    3. Gets the numeric prediction
    4. Applies threshold to het the binary prediction
    5. returns prediction is numeric and binary format
    
    '''
    return 0


