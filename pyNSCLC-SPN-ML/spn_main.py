# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:54:16 2023

@author: japostol
"""

import sys
sys.path.append('C:\\Users\\japostol\\Desktop\\SPN Clinical Python codes\\')
import pandas as pd

import scikitplot as skplt
import spn_clinical_models
import spn_model_evaluation_plots
import matplotlib.pyplot as plt

import spn_clinical_functions

# Load Data
csv_file_path = 'C:\\Users\\japostol\\Desktop\\SPN Clinical Python codes\\labels.xlsx'  # Replace 'your_file.csv' with the actual file path
data = pd.read_excel(csv_file_path)

# Delete the ID, not needed
data=data.drop('ID',axis=1)

# Select the biopsy and delete the column
biopsy_data = data[data.iloc[:, -1] == 1]
biopsy_data = biopsy_data.iloc[:, :-1]

# Select the follow-up and delete the column
follow_data = data[data.iloc[:, -1] == 2]
follow_data = follow_data.iloc[:, :-1]

# Select the doctor and delete the column
doctor_data = data[data.iloc[:, -1] == 3]
doctor_data = doctor_data.iloc[:, :-1]



###############################
#
#
# SOS! YOU NEED TO SPECIFY FEATURE SELECTION
# SPECIFY WHICH FEATURES ARE TO BE REMOVED IN THE MAIN PAGE
# THEN CREATE THE FEATURE_REMOVE FUNCTION
# THEN CALL THIS FUNCTION IN THE TRAIN, FIT ETC FUNCTION
# 
#
#
#
###################################

# FEATURES

selected_features = ['GLU','SUV','DIAMETER','LOCATION_Lingula','TYPE_Semi-solid','MARGINS_lobulated','MARGINS_spiculated','MARGINS_well defined']

# selected_features = ['GLU','SUV','DIAMETER','LOCATION_Lingula','TYPE_Semi-solid','MARGINS_lobulated','MARGINS_spiculated','MARGINS_well defined',
#                       'AGE','LOCATION_Left_Lower_Lobe','LOCATION_Right_Lower_Lobe','TYPE_Consolidated','TYPE_Solid']

# MERGE TWO DATASETS

#doctor_data = pd.concat([doctor_data,follow_data],axis=0)
# follow_data = pd.concat([follow_data,biopsy_data],axis=0)
biopsy_data = pd.concat([follow_data,biopsy_data],axis=0)

#%%
# Assuming the last column is the target variable and the rest are features
X = doctor_data.iloc[:, :-1]  # Features
y = doctor_data.iloc[:, -1]   # Target variable
test = X

AVAILABLE_CLASSIFIERS = ['catboost','logistic','bayes','knn','rf','xgb','lightgbm','svm','nn','adaboost','lda']
# excluded for now NN
classifier_name = 'xgb'

#%% GRID SEARCH

grid_search_model = spn_clinical_functions.grid_search(classifier_name,X,y,test,selected_features)


#%%
# 10F
all_predictions,all_true_labels,classifier,Xen,yen,all_predictions_proba = spn_clinical_functions.train_kfold (classifier_name,X,y,selected_features)
import numpy as np
all_predictions_proba = np.array(all_predictions_proba)
kfold_metrics = spn_clinical_functions.metrics(all_predictions,all_true_labels,all_predictions_proba[:,1])
spn_clinical_functions.print_metrics (kfold_metrics)

# PLOT THE LEARNING CURVE
classifier_untrained = spn_clinical_models.selector(classifier_name='xgb')
spn_model_evaluation_plots.plot_learning_curve_scikit(classifier_untrained, Xen, yen)


# PLOT THE ROC

spn_model_evaluation_plots.plot_roc_scikit(all_predictions_proba,all_true_labels)


# PLOT KS
spn_model_evaluation_plots.plot_ks_statistic(all_true_labels,all_predictions_proba)


# RELIABILITY CURVE FOR ONE
clf_names = ['XGBoost']
prediction_list = [all_predictions_proba]
trues = all_true_labels
spn_model_evaluation_plots.plot_reliability_curve(prediction_list,trues,clf_names)

kappa_score, observed_agreement, expected_agreement = spn_model_evaluation_plots.calculate_cohens_kappa_matrices(all_true_labels, all_predictions)
#%% COMPARISON UNDER K-FOLD FOR DIFFERENT CLASSIFIERS
import numpy as np
COLLECTION_PREDICTIONS = []
COLLECTION_PREDICTIONS_PROBA = []
COLLECTION_LABELS = []
COLLECTION_CLASSIFIERS = ['CatBoost','Naive Bayes','K-NN','Random Forest','XGBoost','LightgbmGBM','SVM','AdaBoost']
classifiers = ['catboost','bayes','knn','rf','xgb','lightgbm','svm','adaboost']

for classifier_name in classifiers:
    all_predictions,all_true_labels,classifier,Xen,yen,all_predictions_proba = spn_clinical_functions.train_kfold (classifier_name,X,y,selected_features)
    COLLECTION_PREDICTIONS.append(all_predictions)
    all_predictions_proba = np.array(all_predictions_proba)
    COLLECTION_PREDICTIONS_PROBA.append(all_predictions_proba)
    COLLECTION_LABELS.append(all_true_labels)

spn_model_evaluation_plots.plot_roc_scikit_multiple_classifiers(COLLECTION_PREDICTIONS_PROBA, COLLECTION_LABELS, COLLECTION_CLASSIFIERS)

spn_model_evaluation_plots.plot_reliability_curve(COLLECTION_PREDICTIONS_PROBA,all_true_labels,COLLECTION_CLASSIFIERS)



#%%
# FIT
all_predictions,all_true_labels,classifier,importance,Xen,y_new,all_predictions_proba = spn_clinical_functions.fit(classifier_name,X,y,test,selected_features)
import numpy as np
all_predictions_proba = np.array(all_predictions_proba)
fit_metrics = spn_clinical_functions.metrics(all_predictions,all_true_labels,all_predictions_proba[:,1])
spn_clinical_functions.print_metrics (fit_metrics)

if classifier_name == 'xgb':
    importance = classifier.feature_importances_
    features = Xen.columns
    IMPORTANCES = pd.DataFrame(importance, index=features)


# PLOT THE FEATURE IMPORRTANCE
trained_classifier = classifier
spn_model_evaluation_plots.plot_feature_importance(trained_classifier,feature_names=selected_features)


# PLOT THE ROC

spn_model_evaluation_plots.plot_roc_scikit(all_predictions_proba,all_true_labels)


# PLOT KS
spn_model_evaluation_plots.plot_ks_statistic(all_true_labels,all_predictions_proba)


# RELIABILITY CURVE FOR ONE
clf_names = ['XGBoost']
prediction_list = [all_predictions_proba]
trues = all_true_labels
spn_model_evaluation_plots.plot_reliability_curve(prediction_list,trues,clf_names)


# COHENS
kappa_score, observed_agreement, expected_agreement = spn_model_evaluation_plots.calculate_cohens_kappa_matrices(all_true_labels, all_predictions)

#%% Test phase
from sklearn.preprocessing import LabelEncoder

# make the X,y
X_follow = follow_data.iloc[:, :-1]
y_follow = follow_data.iloc[:, -1]   # Target variable


# encode
classifiers_needing_label_encoding = ['catboost','logistic','bayes','knn','rf','xgb','lightgbm','svm','nn','adaboost','gmm','lda','elastic'] 
if classifier_name in classifiers_needing_label_encoding:
    # Label encode the target variable y if it contains non-numerical values
    label_encoder = LabelEncoder()
    yen_follow = label_encoder.fit_transform(y_follow)
    
    # Convert non-numerical columns in X to numerical using one-hot encoding
    Xen_follow = pd.get_dummies(X_follow, columns=X_follow.select_dtypes(include=['object']).columns)
    
    # Ensure that X_external_encoded has the same columns as X_encoded in the same order
    columns_order = Xen.columns
    Xen_follow = Xen_follow.reindex(columns=columns_order, fill_value=0)
else:
    Xen_follow = X_follow
    yen_follow = y_follow

Xen_follow = spn_clinical_functions.select_features(Xen_follow,selected_features)

predictions_followup = classifier.predict(Xen_follow)
predictions_followup_proba = classifier.predict_proba(Xen_follow)
follow_metrics = spn_clinical_functions.metrics(predictions_followup,yen_follow,predictions_followup_proba[:,1])
spn_clinical_functions.print_metrics (follow_metrics)

#%% BIOPSY

from sklearn.preprocessing import LabelEncoder

# make the X,y
X_biopsy = biopsy_data.iloc[:, :-1]
y_biopsy = biopsy_data.iloc[:, -1]   # Target variable
classifiers_needing_label_encoding = ['catboost','logistic','bayes','knn','rf','xgb','lightgbm','svm','nn','adaboost','gmm','lda','elastic'] 
if classifier_name in classifiers_needing_label_encoding:
    # Label encode the target variable y if it contains non-numerical values
    label_encoder = LabelEncoder()
    yen_biopsy = label_encoder.fit_transform(y_biopsy)
    
    # Convert non-numerical columns in X to numerical using one-hot encoding
    Xen_biopsy = pd.get_dummies(X_biopsy, columns=X_biopsy.select_dtypes(include=['object']).columns)
    
    # Ensure that X_external_encoded has the same columns as X_encoded in the same order
    columns_order = Xen.columns
    Xen_biopsy = Xen_biopsy.reindex(columns=columns_order, fill_value=0)
else:
    Xen_biopsy = X_biopsy
    yen_biopsy = y_biopsy
Xen_biopsy = spn_clinical_functions.select_features(Xen_biopsy,selected_features)

predictions_biopsy = classifier.predict(Xen_biopsy)
predictions_biopsy_proba = classifier.predict_proba(Xen_biopsy)
biopsy_metrics = spn_clinical_functions.metrics(predictions_biopsy,yen_biopsy,predictions_biopsy_proba[:,1])
spn_clinical_functions.print_metrics (biopsy_metrics)



# PLOT THE ROC
import numpy as np
all_predictions_proba = np.array(predictions_biopsy_proba)
spn_model_evaluation_plots.plot_roc_scikit(all_predictions_proba,yen_biopsy)


# PLOT KS
spn_model_evaluation_plots.plot_ks_statistic(yen_biopsy,predictions_biopsy_proba)


# RELIABILITY CURVE FOR ONE
clf_names = ['XGBoost']
prediction_list = [predictions_biopsy_proba]
trues = yen_biopsy
spn_model_evaluation_plots.plot_reliability_curve(prediction_list,trues,clf_names)


# COHENS
kappa_score, observed_agreement, expected_agreement = spn_model_evaluation_plots.calculate_cohens_kappa_matrices(yen_biopsy, predictions_biopsy)


#%% BIOPSY FOR MULTIPLE CLASSIFIERS

import numpy as np
COLLECTION_PREDICTIONS = []
COLLECTION_PREDICTIONS_PROBA = []
COLLECTION_LABELS = []
COLLECTION_CLASSIFIERS = ['CatBoost','Naive Bayes','K-NN','Random Forest','XGBoost','LightgbmGBM','SVM','AdaBoost']
classifiers = ['catboost','bayes','knn','rf','xgb','lightgbm','svm','adaboost']

for classifier_name in classifiers:
    _,_,classifier,_,_,_,_ = spn_clinical_functions.fit(classifier_name,X,y,test,selected_features)
    predictions_biopsy = classifier.predict(Xen_biopsy)
    predictions_biopsy_proba = classifier.predict_proba(Xen_biopsy)
    predictions_biopsy_proba = np.array(predictions_biopsy_proba)
    COLLECTION_PREDICTIONS.append(predictions_biopsy)
    COLLECTION_PREDICTIONS_PROBA.append(predictions_biopsy_proba)
    COLLECTION_LABELS.append(yen_biopsy)

spn_model_evaluation_plots.plot_roc_scikit_multiple_classifiers(COLLECTION_PREDICTIONS_PROBA, COLLECTION_LABELS, COLLECTION_CLASSIFIERS)

spn_model_evaluation_plots.plot_reliability_curve(COLLECTION_PREDICTIONS_PROBA,yen_biopsy,COLLECTION_CLASSIFIERS)






