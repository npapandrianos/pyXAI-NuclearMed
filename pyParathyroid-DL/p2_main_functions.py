# -*- coding: utf-8 -*-

''' LOAD BASIC LIBRARIRES'''

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import cv2
from PIL import Image 
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import tensorflow as tf
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
import pickle
import random
from scipy import ndimage, misc
from tensorflow.keras.callbacks import EarlyStopping


''' LOAD CUSTOM MODULES'''

from p2_model_maker import make_lvgg, make_xception, make_inception, make_vgg, make_resnet, make_mobile, make_dense, make_eff,make_self,make_multi_self, make_3_vggs, make_3d_bench,cnn_lstm,  make_3_vggs_lstm, make_lb_cnn,make_lb_multi, make_lb_multi_2_img, make_2_vggs,siamese_triplet,siamese_network



def model_save_load(data,labels,epochs,batch_size, model, in_shape, tune, classes,n_split,augmentation,verbose):
    
    if model == 'lvgg':
        model3 = make_lvgg(in_shape, tune, classes)
    elif model == 'xception':
        model3 = make_xception(in_shape, tune, classes)
    elif model == 'vgg':
        model3 = make_vgg(in_shape, 20, classes)
    elif model == 'inception':
        model3 = make_inception(in_shape, tune, classes)
    elif model == 'resnet':
        model3 = make_resnet(in_shape, 20, classes)
    elif model == 'mobile':
        model3 = make_mobile(in_shape, tune, classes)
    elif model == 'dense':
        model3 = make_dense(in_shape, 20, classes)
    elif model == 'efficient':
        model3 = make_eff(in_shape, 20, classes)

    model3.fit(data, labels, epochs=epochs, batch_size=batch_size)
    model3.save('C:\\Users\\User\\{}.h5'.format(model))
    loaded_trained_model = tf.keras.models.load_model('C:\\Users\\User\\{}.h5'.format(model))
    
    
    return loaded_trained_model


def add_noise(img):
    '''Add random noise to an image'''
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    img = np.random.poisson(img * vals) / float(vals)
    
    return img


def threeD_augmentor(data,labels):
    
    for omega in range (len(data)):
        stack = data[omega,:,:,:]
        label = labels[omega,:].reshape(1,-1)
        
        if label[0,0] == 1:
            new_imgs = []
            for i in range (stack.shape[0]):
                img = stack[i,:,:,:]
                rot = ndimage.rotate(img, 10, reshape=False)
                new_imgs.append(rot)
            
            new_3d = np.stack(new_imgs,axis=0)
            new_3d = new_3d[None]
            data = np.concatenate((data,new_3d))
            labels = np.concatenate((labels,label))
            
            new_imgs = []
            for i in range (stack.shape[0]):
                img = stack[i,:,:,:]
                rot = ndimage.rotate(img, 20, reshape=False)
                new_imgs.append(rot)
            
            new_3d = np.stack(new_imgs,axis=0)
            new_3d = new_3d[None]
            data = np.concatenate((data,new_3d))
            labels = np.concatenate((labels,label))
            
            new_imgs = []
            for i in range (stack.shape[0]):
                img = stack[i,:,:,:]
                rot = ndimage.shift(img, 8, cval=0)
                new_imgs.append(rot)
            
            new_3d = np.stack(new_imgs,axis=0)
            new_3d = new_3d[None]
            data = np.concatenate((data,new_3d))
            labels = np.concatenate((labels,label))
            
            new_imgs = []
            for i in range (stack.shape[0]):
                img = stack[i,:,:,:]
                rot = ndimage.shift(img, -8, cval=0)
                new_imgs.append(rot)
            
            new_3d = np.stack(new_imgs,axis=0)
            new_3d = new_3d[None]
            data = np.concatenate((data,new_3d))
            labels = np.concatenate((labels,label))
        else:
            new_imgs = []
            for i in range (stack.shape[0]):
                img = stack[i,:,:,:]
                rot = ndimage.rotate(img, 10, reshape=False)
                new_imgs.append(rot)
            
            new_3d = np.stack(new_imgs,axis=0)
            new_3d = new_3d[None]
            data = np.concatenate((data,new_3d))
            labels = np.concatenate((labels,label))
                  
            
            new_imgs = []
            for i in range (stack.shape[0]):
                img = stack[i,:,:,:]
                rot = ndimage.shift(img, -8, cval=0)
                new_imgs.append(rot)
            
            new_3d = np.stack(new_imgs,axis=0)
            new_3d = new_3d[None]
            data = np.concatenate((data,new_3d))
            labels = np.concatenate((labels,label))
    
    return data,labels
    
    
from tensorflow import keras

def plot_confusion_matrix(cm, classes,
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
    
    

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def feature_maps(path,save_path,model,in_shape):
    
    from matplotlib import pyplot
    from numpy import expand_dims
    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    
    h = in_shape[1]
    w = in_shape[0]
    import os
    from imutils import paths
    
    imagePaths = sorted(list(paths.list_images(path)))
    preprocess_input = keras.applications.mobilenet.preprocess_input
    
    for imagePath in imagePaths[:5]:
        name = os.path.splitext(os.path.basename(imagePath))[0]
        ext = os.path.splitext(os.path.basename(imagePath))[1]
    # redefine model to output right after the first hidden layer
        ixs = [2, 5, 9, 13, 17]
        outputs = [model.layers[i].output for i in ixs]
        model2 = tf.keras.Model(inputs=model.inputs, outputs=outputs)
        # load the image with the required shape
        
        
        img =get_img_array(imagePath, size=(h, w))
        # expand dimensions so that it represents a single 'sample'
        # prepare the image (e.g. scale pixel values for the vgg)
        # img = preprocess_input(img)
        # get feature map for first hidden layer
        feature_maps = model2.predict(img)
        # plot the output from each block
        square = 3
        o = 1
        for fmap in feature_maps:
         	# plot all 64 maps in an 8x8 squares
             ix = 1
             for _ in range(square):
                 for _ in range(square):
                     ax = pyplot.subplot(square, square, ix)
                     ax.set_xticks([])
                     ax.set_yticks([]); fig = pyplot.gcf();
                     pyplot.imshow(fmap[0, :, :, ix-1], cmap='gist_gray')
                     ix += 1
             o=o+1 ; p = save_path + name + str(o) + ext ; fig.savefig(p)
    return imagePath



'''MULTI 2 IMGS'''

def train_multi(data,labels,epochs,batch_size, model, in_shape, tune, classes, n_split,augmentation,verbose,logs,plot_results,class_names,save_variables,save_model_just=False):
    import warnings
    warnings.filterwarnings("ignore")
    model_in_shape = (in_shape[1],in_shape[0],in_shape[2])
    
    #labels = tf.keras.utils.to_categorical(labels, num_classes=classes)
    
    #early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.05, patience = 20, mode='max', restore_best_weights=True)
    
    if augmentation:
        thrs = 0.9
    else:
        thrs = 0.98
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            #print (epoch)
            if logs.get('accuracy') >= thrs:
                self.model.stop_training = True
                logging.warning('Stopped at epoch: {}'.format(epoch))
    
    callback = CustomCallback()    
    # CONFIGURE LOGGINGS

    logging.basicConfig(filename='train_logger.log', level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
    my_string = ','.join(class_names)
    
    params = {"Epochs:": epochs,
              "batch_size:": batch_size,
              "Model:": model,
              "In_shape:": in_shape,
              "Folds:": n_split,
              "Tune:": tune,
              "Classes:": classes,
              "Class_names:": my_string,
              "Augmentations:": augmentation,
        }
    
    if logs:
        logging.info ("==============================Started New Training with {} ==============================".format(model))
        logging.info ("Params: {}".format(params))
        



    # INITIATE SCORE LISTS AND TABLES

    scores = [] #here every fold accuracy will be kept
    f1_scores = []
    recalls = []
    precisions = []
    predictions_all = np.empty(0) # here, every fold predictions will be kept
    test_labels = np.empty(0) #here, every fold labels are kept
    conf_final = np.array([[0,0],[0,0]])
    predictions_all_num = np.empty([0,classes])
    auc_scores = []
    fold_metrics_all = []



    # INITIATE VARIABLE 

    omega = 1

    # INITIATE FOLD CR0SS - VALIDATION  
    
    
    
    if save_model_just:
        
        trainX = data
        trainY = labels
        import datetime
        now = datetime.datetime.now()
        now = now.strftime("%Y_%m_%d_%H_%M_%S")
        name = 'saved_{}_{}'.format(model,now)
        model3 = make_lvgg(in_shape, tune, classes)
        
        if augmentation:
            print ('Activating Augmentations')
            aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,width_shift_range=5)
            history = model3.fit_generator(aug.flow(trainX, trainY,batch_size=batch_size), epochs=epochs,callbacks=[callback], steps_per_epoch=len(trainX)//batch_size)
        else:
            history = model3.fit(trainX, trainY, validation_split=0.1, epochs=epochs, batch_size=batch_size,callbacks=[callback])
        
        #model3.save('C:\\Users\\User\\{}.h5'.format(name))
        
        group_results = fold_scores = pd_metrics = predictions_all = predictions_all_num = test_labels = conf_final = cnf_matrix = conf_img = None
        
        return model3, group_results , fold_scores , pd_metrics ,predictions_all ,predictions_all_num ,test_labels ,labels ,conf_final ,cnf_matrix ,conf_img ,history
    
    
    
    
    
    for train_index, test_index in KFold(n_split).split(data):
        trainX, testX = data[train_index], data[test_index]
        trainY, testY = labels[train_index], labels[test_index]



        # BUILD MODELS FOR EVERY FOLD

        if model == 'lvgg':
            model3 = make_lvgg(in_shape, tune, classes)
        elif model == 'inception':
            model3 = make_inception(in_shape, tune, classes)
        elif model == 'vgg':
            model3 = make_vgg(in_shape, 20, classes)
        elif model == 'xception':
            model3 = make_xception(in_shape, tune, classes)
        elif model == 'resnet':
            model3 = make_resnet(in_shape, 20, classes)
        elif model == 'mobile':
            model3 = make_mobile(in_shape, tune, classes)
        elif model == 'dense':
            model3 = make_dense(in_shape, 20, classes)
        elif model == 'efficient':
            model3 = make_eff(in_shape, 20, classes)
        # UTILITIES

        if omega == 1:
            model3.summary()
        omega = omega + 1

        if verbose:
            print('-------------- PREPARING FOLD: {} -----------------------'.format(omega-1))
        if logs:
            logging.info('-------------- PREPARING FOLD: {} -----------------------'.format(omega-1))



        # TRAINING OPERATION
        
        if augmentation:
            print ('Activating Augmentations')
            aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=5,width_shift_range=4)
            history = model3.fit_generator(aug.flow(trainX, trainY,batch_size=batch_size), epochs=epochs,callbacks=[callback], steps_per_epoch=len(trainX)//batch_size)
        else:

            history = model3.fit(trainX, trainY, validation_split=0.1, epochs=epochs, batch_size=batch_size,callbacks=[callback])
        
        
        
        # VALIDATION OPERATION
        
        predict = model3.predict(testX) #for def models functional api
        predict_num = predict
        predict = predict.argmax(axis=-1) #for def models functional api
        
        score = model3.evaluate(testX,testY)
        score = score[1] #keep the accuracy score, not the loss
        scores.append(score) #put the fold score to list
        testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
        #print(testY2)

        if classes == 2:
            recall = recall_score(testY2,predict)
            recalls.append(recall)
            
            precision = precision_score(testY2,predict)
            precisions.append(precision)
        
            oneclass = predict_num[:,1].reshape(-1,1)
            #print(oneclass)
            aucS = roc_auc_score(testY2, oneclass)
            auc_scores.append(aucS)
            conf = confusion_matrix(testY2, predict) #get the fold conf matrix
            conf_final = conf + conf_final
        
            f1 = f1_score(testY2, predict)
            f1_scores.append(f1)
            
            average_precision = 'n/a'
        else:
            precision = dict()
            recall = dict()
            average_precision = dict()
            for i in range(classes):
                precision[i], recall[i], _ = precision_recall_curve(testY[:, i], predict_num[:, i])
                average_precision[i] = average_precision_score(testY[:, i], predict_num[:, i])

            # A "micro-average": quantifying score on all classes jointly
            precision["micro"], recall["micro"], _ = precision_recall_curve(testY.ravel(), predict_num.ravel())
            average_precision["micro"] = average_precision_score(testY, predict_num, average="micro")
            
        FP = conf[0,1]
        FN = conf[1,0]
        TP = conf[1,1]
        TN = conf[0,0]
        
        specificity_fold = TN/(TN+FP)
        NPV_fold = TN/(TN+FN)
        sensitivity_fold = TP/(TP+FN)
        PPV_fold = TP/(TP+FP)
    
        predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
        predictions_all_num = np.concatenate([predictions_all_num, predict_num])
        testY = testY.argmax(axis=-1)
        test_labels = np.concatenate ([test_labels, testY]) #merge the two np arrays of labels#
        
        fold_metrics = {
        "AUC Score": aucS,
        "Accuracy" : score,
        "F1 Score" : f1,
        "Precision": precision,
        "REcall" : recall,
        "Avg Precisions" : average_precision,
        "Sensitivity" : sensitivity_fold,
        "Specificity" : specificity_fold,
        "PPV" :PPV_fold,
        "NPV" : NPV_fold,
        "TP" : TP,
        "FP" : FP,
        "TN" : TN,
        "FN" :FN           
        }
        
        fold_metrics_all.append(fold_metrics)

        # UTILITES

        if verbose:
            for key, value in fold_metrics.items():
                print(key, ' : ', value)
        if logs:
            logging.info(str(fold_metrics))
        
    
    
    
    # AFTER-FOLD EVALUATION METRICS

    aucS = roc_auc_score(labels, predictions_all_num)
    rounded_labels = np.argmax(labels, axis=1)
    conf_final = confusion_matrix(rounded_labels, predictions_all)
    
    if classes == 2:
        precision_final = precision_score(np.argmax(labels,axis=-1),np.argmax(predictions_all_num,axis=-1))
        recall_final = recall_score(np.argmax(labels,axis=-1),np.argmax(predictions_all_num,axis=-1))
    else:
        precision_final = 'n/a'
        recall_final = 'n/a'
    
    
    scores = np.asarray(scores)
    final_score = np.mean(scores)
    f1sc = np.asarray(f1_scores)
    mean_f1 = np.mean(f1sc)
    
    FP = conf_final[0,1]
    FN = conf_final[1,0]
    TP = conf_final[1,1]
    TN = conf_final[0,0]
        
    specificity = TN/(TN+FP)
    NPV = TN/(TN+FN)
    sensitivity = TP/(TP+FN)
    PPV = TP/(TP+FP)
    
    
    
    # PLOT OPERATIONS
    
    if plot_results:
    
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
        
        for i,item in enumerate( class_names ):
            fpr[i], tpr[i], _ = roc_curve(labels[:, i],predictions_all_num[:, i])
            plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(item))
            plt.plot(fpr[i], tpr[i], lw=2)
        
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.legend(loc="best")
        plt.title("ROC curve")
        plt.grid(False)
        plt.gca().spines['bottom'].set_color('0.5')
        plt.gca().spines['top'].set_color('0.5')
        plt.gca().spines['right'].set_color('0.5')
        plt.gca().spines['left'].set_color('0.5')
        plt.savefig('C:\\Users\\User\\rocurves.png', dpi=300)
        plt.show()
    
    
    
    
    # CONFUSION MATRIX OPERATIONS
    
        
    cnf_matrix = confusion_matrix(np.argmax(labels,axis=-1), predictions_all)
    
    
    
    
        #np.set_printoptions(precision=2)
    
    conf_img = plot_confusion_matrix(cnf_matrix, class_names)
        #conf_img = 0

    metrics = {
        "AUC Score": aucS,
        "Accuracy" : final_score,
        "F1 Score" : mean_f1,
        "Precision": precision_final,
        "REcall" : recall_final,
        "Avg Precisions" : average_precision,
        "Sensitivity" : sensitivity,
        "Specificity" :specificity,
        "PPV" : PPV,
        "NPV" : NPV,
        "TP" : TP,
        "FP" : FP,
        "TN" : TN,
        "FN" :FN      
        }
    
    import pandas as pd
    pd_metrics = pd.DataFrame.from_dict(metrics,orient='index')
    if verbose:
        for key, value in metrics.items():
                print(key, ' : ', value)
                
    if logs:
        logging.info("------------------FOLDS FINISHED--------------------")
        logging.info(str(metrics))
                
                
    group_results = [pd_metrics,predictions_all,predictions_all_num,test_labels,labels,conf_final,cnf_matrix,conf_img,history]
    
    fold_scores = [scores,f1sc,recalls,precisions,auc_scores]
    
    if save_variables:
        pass
        #with open('variables.pkl', 'wb') as f:  
            #pickle.dump([group_results, fold_scores], f)
    
    return model3, group_results, fold_scores, pd_metrics,predictions_all,predictions_all_num,test_labels,labels,conf_final,cnf_matrix,conf_img,history