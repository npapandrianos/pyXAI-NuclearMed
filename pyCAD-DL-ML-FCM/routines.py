# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:07:24 2022

@author: John
"""
import time
import logging
import numpy as np
import sklearn
import os
import json

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt
plt.style.use('ggplot')


import models
import custom_3d_cnn



def COMMUNICATE (message,log_it=True,print_it=True,line=False):
    if log_it:
        logging.info(message)
    if print_it:
        if line:
            print ("")
        print (message)


def txt_export(variable,name, type_of_variable,save_path):
    name = name + '.txt'
    full = os.path.join(save_path,name)    
    if type_of_variable == 'np':
        np.savetxt(full,variable,fmt="%s") #fmt='%10.5f
    elif  type_of_variable == 'dic':
        with open(full, 'w') as file:
            file.write(json.dumps(variable))
    elif  type_of_variable == 'df':
        variable.to_csv(r'{}'.format(full))
    elif  type_of_variable == 'fl':    
        with open(full, 'w') as f:
            f.write(str(variable))
            f.close()
            

def exporter (list_of_vars, save_path):
    
    try:
        del list_of_vars[3]['CNF']
        del list_of_vars[4]['CNF']
    except Exception:
        pass
    
    keys_values = list_of_vars[3].items()
    new_d = {str(key): str(value) for key, value in keys_values}
    list_of_vars[3] = new_d

    keys_values = list_of_vars[4].items()
    new_d = {str(key): str(value) for key, value in keys_values}
    list_of_vars[4] = new_d  
    
    keys_values = list_of_vars[15].items()
    new_d = {str(key): str(value) for key, value in keys_values}
    list_of_vars[15] = new_d    
    
    list_of_names = ['ATT','dataset','expert_labels','final_metrics','final_image_metrics','folds_metrics','folds_metrics_imgs',
                     'info','labels','predictions','predictions_image','num_predictions','num_predictions_image',
                     'test_labels','duration','Agreements']
    
    list_of_types = ['df','np','np','dic','dic','lis','li',
                     'df','np','np','np','np','np',
                     'np','fl','dic']
    
    for i in range (len(list_of_vars)):
        txt_export (list_of_vars[i],list_of_names[i],list_of_types[i],save_path)
    


def pick_model (OPTIONS_MODE,OPTIONS_PREPROCESSING,OPTIONS_TRAINING):
    
    if OPTIONS_MODE['cnn'] == 'vgg19':
        model = models.make_vgg19(OPTIONS_MODE,OPTIONS_PREPROCESSING,OPTIONS_TRAINING)
    if OPTIONS_MODE['cnn'] == '3d_main_70':
        model = custom_3d_cnn.make_3d_main_70_emerald(OPTIONS_MODE,OPTIONS_PREPROCESSING,OPTIONS_TRAINING)
    if OPTIONS_MODE['cnn'] == '3d_main_exp':
        model = custom_3d_cnn.make_3d_main_70_emerald_exp(OPTIONS_MODE,OPTIONS_PREPROCESSING,OPTIONS_TRAINING) 
    if OPTIONS_MODE['cnn'] == 'cnn_volume':
        model = custom_3d_cnn.make_volume_all(OPTIONS_MODE,OPTIONS_PREPROCESSING,OPTIONS_TRAINING)       
        
    return model

def pick_model_clinical (OPTIONS_MODE,OPTIONS_PREPROCESSING,OPTIONS_TRAINING):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn import svm
    from sklearn.linear_model import SGDClassifier
    from sklearn import tree
    from sklearn.ensemble import BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    
    
    
    
    if OPTIONS_MODE['classifier'] == 'rf':
        if OPTIONS_MODE['grid_search']:
            print ('Selected Classifier RF with Grid Search. Grid search will be applied only to the first fold')
            from sklearn.model_selection import GridSearchCV
            param_grid = { 
                            'n_estimators': [200,300,400,500,700,1000],
                            'max_depth' : [4,8,12],
                            'min_samples_split': [1,2,4],
                            'min_samples_leaf': [1,2,4],
                            }
            
            RF = RandomForestClassifier(random_state=42)
            model_ml = GridSearchCV(estimator=RF, param_grid=param_grid, cv= 10)
        else:
            model_ml = RandomForestClassifier(n_estimators = 500,max_depth=8,min_samples_split=2,min_samples_leaf=4)
            
            
    if OPTIONS_MODE['classifier'] == 'ada':
        model_ml = AdaBoostClassifier(n_estimators=50,learning_rate=1)
    if OPTIONS_MODE['classifier'] == 'svm':
        model_ml = svm.SVC()
    if OPTIONS_MODE['classifier'] == 'sgd':
        model_ml = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    if OPTIONS_MODE['classifier'] == 'tree':
        model_ml = tree.DecisionTreeClassifier()       
    if OPTIONS_MODE['classifier'] == 'bag_knn':
        model_ml = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)      
    if OPTIONS_MODE['classifier'] == 'mlp':    
        model_ml = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    if OPTIONS_MODE['classifier'] == 'FCM':    
        model_ml = 0
        
    return model_ml    
    
def metrics (predict,predict_num,test_label, test_label_onehot):
    # return any metrics
    start = time.time()
    recall = recall_score(test_label_onehot,predict)
    precision = precision_score(test_label_onehot,predict)
    oneclass = predict_num[:,1].reshape(-1,1)
    #print(oneclass)
    auc = roc_auc_score(test_label_onehot, oneclass)
    conf = confusion_matrix(test_label_onehot, predict) #get the fold conf matrix
    f1 = f1_score(test_label_onehot, predict)
      
    FP = conf[0,1]
    FN = conf[1,0]
    TP = conf[1,1]
    TN = conf[0,0]
    
    specificity = TN/(TN+FP)
    NPV = TN/(TN+FN)
    sensitivity = TP/(TP+FN)
    PPV = TP/(TP+FP)
    acc = (TP+TN)/(TP+TN+FP+FN)
    
    metrics = {
    "AUC": auc,
    "Accuracy" : acc,
    "F1" : f1,
    "Precision": precision,
    "Recall" : recall,
    "Sensitivity" : sensitivity,
    "Specificity" : specificity,
    "PPV" :PPV,
    "NPV" : NPV,
    "TP" : TP,
    "FP" : FP,
    "TN" : TN,
    "FN" :FN,
    "CNF": conf         
    }

    end = time.time()
    duration = (end - start)
    return metrics
    

def metrics_agreement(expert_labels,predictions):
    experts = expert_labels.argmax(axis=-1)
    ck = sklearn.metrics.cohen_kappa_score(experts,predictions)
    conf = confusion_matrix(experts, predictions)
    FP = conf[0,1]
    FN = conf[1,0]
    TP = conf[1,1]
    TN = conf[0,0]
    aggr = (TP+TN)/(TP+TN+FP+FN)
    
    metrics = {
        'Agreement':aggr,
        "TP" : TP,
        "FP" : FP,
        "TN" : TN,
        "FN" :FN,
        'Cohens' : ck
        }
    
    return metrics
    
    
    
    
def history_plots (history,class_names,predictions_all_num,labels):
    
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
# summarize history for accuracy
    plt.plot(history.history['accuracy'])
    
    if 'val_acc' in history.history.keys():
        plt.plot(history.history['val_acc'])
    elif 'val_accuracy' in history.history.keys():
        plt.plot(history.history['val_accuracy'])

    plt.title('Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    
    if 'val_acc' in history.history.keys():
        plt.legend(['train', 'validation'], loc='upper left')
    if 'val_accuracy' in history.history.keys():
        plt.legend(['train', 'validation'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    plt.grid(False)
    plt.gca().spines['bottom'].set_color('0.5')
    plt.gca().spines['top'].set_color('0.5')
    plt.gca().spines['right'].set_color('0.5')
    plt.gca().spines['left'].set_color('0.5')
    plt.savefig('C:\\Users\\User\\accs.png', dpi=300)
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    if 'val_loss' in history.history.keys():
        plt.plot(history.history['val_loss'])

    plt.title('Losses')
    plt.ylabel('Loss (%)')
    plt.xlabel('Epoch')
    if 'val_loss' in history.history.keys():
        plt.legend(['train', 'validation'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    plt.grid(False)
    plt.gca().spines['bottom'].set_color('0.5')
    plt.gca().spines['top'].set_color('0.5')
    plt.gca().spines['right'].set_color('0.5')
    plt.gca().spines['left'].set_color('0.5')
    plt.savefig('C:\\Users\\User\\losses.png', dpi=300)
    plt.show()
    
    # roc curve
    fpr = dict()
    tpr = dict()
    
    colorp = ['red','black']
    for i,item in enumerate( class_names ):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i],predictions_all_num[:, i])
        plt.plot(fpr[i], tpr[i], lw=1, label='class {}'.format(item),color=colorp[i])
        plt.plot(fpr[i], tpr[i], lw=1,color=colorp[i])
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="best")
    plt.title("ROC curve")
    plt.grid(False)
    plt.gca().spines['bottom'].set_color('0.5')
    plt.gca().spines['top'].set_color('0.5')
    plt.gca().spines['right'].set_color('0.5')
    plt.gca().spines['left'].set_color('0.5')
    plt.savefig('C:\\Users\\User\\rocurves.png', dpi=300)
    plt.show()    
    
def conf_matrix_plot (cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    
   #  fig, ax = plt.subplots(figsize=(15,15))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=70, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.xlim(-0.5, len(classes)-0.5) # ADD THIS LINE
    plt.ylim(len(classes)-0.5, -0.5) # ADD THIS LINE
    fig.tight_layout()
    plt.grid(False)
    plt.show()
    fig.savefig('C:\\Users\\User\\cnf.png', dpi=300)
    
    start = time.time()
    end = time.time()
    duration = (end - start)
    
    
    
    
def grad_cam ():

    start = time.time()
    end = time.time()
    duration = (end - start)   




    