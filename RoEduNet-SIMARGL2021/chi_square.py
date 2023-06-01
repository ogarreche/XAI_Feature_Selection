#!/usr/bin/env python
# coding: utf-8

print('--------------------------------------------------')
print('--------------------------------------------------')
print('Importing Libraries')
print('--------------------------------------------------')

# Makes sure we see all columns

from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.datasets import load_iris
# Loading Scikits random fo
#rest classifier
import sklearn
#import lime
from sklearn.preprocessing import OneHotEncoder

from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
#from sklearn.metrics import auc_score
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter
from sklearn.preprocessing import label_binarize
#loading pandas
import pandas as pd
#Loading numpy
import numpy as np
# Setting random seed
import time

import shap
from scipy.special import softmax
np.random.seed(0)
import matplotlib.pyplot as plt
import random

from sklearn.metrics import f1_score, accuracy_score
#from interpret.blackbox import LimeTabular
from interpret import show
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
pd.set_option('display.max_columns', None)
shap.initjs()

print('Defining Function')
print('--------------------------------------------------')

def oversample(X_train, y_train):
    oversample = RandomOverSampler(sampling_strategy='minority')
    # Convert to numpy and oversample
    x_np = X_train.to_numpy()
    y_np = y_train.to_numpy()
    x_np, y_np = oversample.fit_resample(x_np, y_np)
    
    # Convert back to pandas
    x_over = pd.DataFrame(x_np, columns=X_train.columns)
    y_over = pd.Series(y_np)
    return x_over, y_over
def print_feature_importances_shap_values(shap_values, features):
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the feature
s, on the order presented to the explainer
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")

def ACC(x,y,w,z):
    Acc = (x+y)/(x+w+z+y)
    return Acc

def PRECISION(x,w):
    Precision = x/(x+w)
    return Precision
def RECALL(x,z):
    Recall = x/(x+z)
    return Recall
def F1(Recall, Precision):
    F1 = 2 * Recall * Precision / (Recall + Precision)
    return F1
def BACC(x,y,w,z):
    BACC =(x/(x+z)+ y/(y+w))*0.5
    return BACC
def MCC(x,y,w,z):
    MCC = (y*x-z*w)/(((x+w)*(x+z)*(y+w)*(y+z))**.5)
    return MCC
def AUC_ROC(y_test_bin,y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_avg = 0
    counting = 0
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
     # plt.plot(fpr[i], tpr[i], color='darkorange', lw=2)
      #print('AUC for Class {}: {}'.format(i+1, auc(fpr[i], tpr[i])))
        auc_avg += auc(fpr[i], tpr[i])
        counting = i+1
    return auc_avg/counting


print('Selecting Column Features')
print('--------------------------------------------------')

# Choose features to import + label
req_cols = ['PROTOCOL_MAP','FLOW_DURATION_MILLISECONDS','TCP_FLAGS','TCP_WIN_MAX_IN',
'TCP_WIN_MAX_OUT','TCP_WIN_MIN_IN','TCP_WIN_MIN_OUT','TCP_WIN_MSS_IN','TCP_WIN_SCALE_IN',
'TCP_WIN_SCALE_OUT','SRC_TOS','DST_TOS','ALERT']

req_cols = ['FLOW_DURATION_MILLISECONDS','FIRST_SWITCHED',
            'TOTAL_FLOWS_EXP','TCP_WIN_MSS_IN','LAST_SWITCHED',
            'TCP_WIN_MAX_IN','TCP_WIN_MIN_IN','TCP_WIN_MIN_OUT',
           'PROTOCOL','TCP_WIN_MAX_OUT','TCP_FLAGS',
            'TCP_WIN_SCALE_OUT','TCP_WIN_SCALE_IN','SRC_TOS',
            'DST_TOS','ALERT']

print('Loading Database')
print('--------------------------------------------------')

#Denial of Service
df0 = pd.read_csv ('sensor_db/dos-03-15-2022-15-44-32.csv', usecols=req_cols)
df1 = pd.read_csv ('sensor_db/dos-03-16-2022-13-45-18.csv', usecols=req_cols)
df2 = pd.read_csv ('sensor_db/dos-03-17-2022-16-22-53.csv', usecols=req_cols)
df3 = pd.read_csv ('sensor_db/dos-03-18-2022-19-27-05.csv', usecols=req_cols)
df4 = pd.read_csv ('sensor_db/dos-03-19-2022-20-01-53.csv', usecols=req_cols)
df5 = pd.read_csv ('sensor_db/dos-03-20-2022-14-27-54.csv', usecols=req_cols)

#Normal
df7 = pd.read_csv  ('sensor_db/normal-03-15-2022-15-43-44.csv', usecols=req_cols)
df8 = pd.read_csv  ('sensor_db/normal-03-16-2022-13-44-27.csv', usecols=req_cols)
df9 = pd.read_csv  ('sensor_db/normal-03-17-2022-16-21-30.csv', usecols=req_cols)
df10 = pd.read_csv ('sensor_db/normal-03-18-2022-19-17-31.csv', usecols=req_cols)
df11 = pd.read_csv ('sensor_db/normal-03-18-2022-19-25-48.csv', usecols=req_cols)
df12 = pd.read_csv ('sensor_db/normal-03-19-2022-20-01-16.csv', usecols=req_cols)
df13 = pd.read_csv ('sensor_db/normal-03-20-2022-14-27-30.csv', usecols=req_cols)

#PortScanning
df14 = pd.read_csv  ('sensor_db/portscanning-03-15-2022-15-44-06.csv', usecols=req_cols)
df15 = pd.read_csv  ('sensor_db/portscanning-03-16-2022-13-44-50.csv', usecols=req_cols)
df16 = pd.read_csv  ('sensor_db/portscanning-03-17-2022-16-22-53.csv', usecols=req_cols)
df17 = pd.read_csv  ('sensor_db/portscanning-03-18-2022-19-27-05.csv', usecols=req_cols)
df18 = pd.read_csv  ('sensor_db/portscanning-03-19-2022-20-01-45.csv', usecols=req_cols)
df19 = pd.read_csv  ('sensor_db/portscanning-03-20-2022-14-27-49.csv', usecols=req_cols)

frames = [df0, df1, df2, df3, df4, df5, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19]

#concat data frames
df = pd.concat(frames,ignore_index=True)

# shuffle the DataFrame rows
df = df.sample(frac = 1)

# assign alert column to y
y = df.pop('ALERT')

# join alert back to df
df = df.assign( ALERT = y) 

#Fill NaN with 0s
df = df.fillna(0)

# Normalize database
print('---------------------------------------------------------------------------------')
print('Normalizing database')
print('---------------------------------------------------------------------------------')
print('')

# make a df copy
df_max_scaled = df.copy()

# assign alert column to y
y = df_max_scaled.pop('ALERT')

#Normalize operation
for col in df_max_scaled.columns:
    t = abs(df_max_scaled[col].max())
    df_max_scaled[col] = df_max_scaled[col]/t
df_max_scaled
#assign df copy to df
df = df_max_scaled.assign( ALERT = y)
#Fill NaN with 0s
df = df.fillna(0)

# Separate features and labels
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['ALERT']), df['ALERT'], test_size=0.2, random_state=42)



# Perform chi-square feature selection
chi_selector = SelectKBest(chi2, k=10) # select top 5 features
X_train_chi = chi_selector.fit_transform(X_train, y_train)
X_test_chi = chi_selector.transform(X_test)

# Print the namesi


# get the indices of the selected features
selected_indices = chi_selector.get_support(indices=True)

# get the names of the selected features
selected_features = list(X_train.columns[selected_indices])

# print the selected features
print('Selected Features:', selected_features)
