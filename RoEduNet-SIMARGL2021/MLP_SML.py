print('---------------------------------------------------------------------------------')
print('MLP')
print('---------------------------------------------------------------------------------')
print('Importing libraries ')
print('---------------------------------------------------------------------------------')

#----------------------------------------------------------------------
#import tensorflow as tf
import os
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from matplotlib import pyplot as plt
import numpy as np
# import pafy
import pandas as pd
#import csv
import math
#import keras
#from keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers import Dropout
#from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
#from keras.callbacks import EarlyStopping
#from keras.preprocessing import sequence
#from keras.utils import pad_sequences
#from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.neural_network import MLPClassifier
import sklearn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import shap
np.random.seed(0)
#----------------------------------------------------------------------
#---------------------------------------------------------------------
print('---------------------------------------------------------------------------------')
print('Defining Metrics')
print('---------------------------------------------------------------------------------')
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

def ACC(TP,TN,FP,FN):
    Acc = (TP+TN)/(TP+FP+FN+TN)
    return Acc

def PRECISION(TP,FP):
    Precision = TP/(TP+FP)
    return Precision
def RECALL(TP,FN):
    Recall = TP/(TP+FN)
    return Recall
def F1(Recall, Precision):
    F1 = 2 * Recall * Precision / (Recall + Precision)
    return F1
def BACC(TP,TN,FP,FN):
    BACC =(TP/(TP+FN)+ TN/(TN+FP))*0.5
    return BACC
def MCC(TP,TN,FP,FN):
    MCC = (TN*TP-FN*FP)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**.5)
    return MCC
def AUC_ROC(y_test_bin,y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_avg = 0
    counting = 0
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        auc_avg += auc(fpr[i], tpr[i])
        counting = i+1
    return auc_avg/counting
#-----------------------------------------------------------------------
#----------------------------------------------------------

print('---------------------------------------------------------------------------------')
print('Defining features')
print('---------------------------------------------------------------------------------')
'''
########################################### SIMARGL Features ########################################
'''

# Select which feature method you want to use by uncommenting it.

'''
all features
'''

'''
req_cols = ['FLOW_DURATION_MILLISECONDS','FIRST_SWITCHED',
            'TOTAL_FLOWS_EXP','TCP_WIN_MSS_IN','LAST_SWITCHED',
            'TCP_WIN_MAX_IN','TCP_WIN_MIN_IN','TCP_WIN_MIN_OUT',
           'PROTOCOL','TCP_WIN_MAX_OUT','TCP_FLAGS',
            'TCP_WIN_SCALE_OUT','TCP_WIN_SCALE_IN','SRC_TOS',
            'DST_TOS','FLOW_ID','L4_SRC_PORT','L4_DST_PORT',
           'MIN_IP_PKT_LEN','MAX_IP_PKT_LEN','TOTAL_PKTS_EXP',
           'TOTAL_BYTES_EXP','IN_BYTES','IN_PKTS','OUT_BYTES','OUT_PKTS',
            'ALERT']

'''


'''
##################################### For K = 15 ################################################
'''

'''
 k gain according CICIDS paper
'''

'''
req_cols =  [ 'FLOW_DURATION_MILLISECONDS', 'FIRST_SWITCHED', 'TOTAL_FLOWS_EXP', 'TCP_WIN_MSS_IN', 'LAST_SWITCHED', 'TCP_WIN_MAX_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_MIN_OUT', 'PROTOCOL', 'TCP_WIN_MAX_OUT','ALERT' ]
'''


'''
##################################### For K = 10 ################################################
'''

'''
 1 - Common features by overall rank
'''

'''

req_cols =  [ 'TCP_WIN_SCALE_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_MSS_IN', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS', 'TCP_WIN_MAX_OUT', 'TCP_WIN_MIN_OUT', 'SRC_TOS', 'DST_TOS','ALERT' ]


'''

'''
 2 - Chi square
'''

'''

req_cols =  [ 'FLOW_DURATION_MILLISECONDS', 'PROTOCOL', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'TCP_WIN_MIN_IN', 'TCP_WIN_MIN_OUT', 'TCP_WIN_SCALE_IN', 'TCP_WIN_SCALE_OUT', 'SRC_TOS', 'DST_TOS','ALERT' ]


'''

'''
 3 - Feature Correlation
'''

'''
req_cols =  [ 'TCP_WIN_MSS_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_SCALE_IN', 'PROTOCOL', 'TCP_WIN_MAX_OUT', 'TCP_WIN_MIN_OUT', 'TCP_WIN_SCALE_OUT', 'FIRST_SWITCHED', 'LAST_SWITCHED','ALERT' ]

'''

'''
 4 - Feature Importance
'''

'''

req_cols =  [ 'TCP_WIN_MSS_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_SCALE_IN', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS', 'TCP_WIN_MAX_OUT', 'TCP_WIN_MIN_OUT', 'TOTAL_FLOWS_EXP', 'LAST_SWITCHED','ALERT' ]

'''

'''
 5 - Models + attacks
'''

'''
req_cols =  [ 'TCP_WIN_MSS_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_SCALE_IN', 'TCP_WIN_MIN_IN', 'FLOW_DURATION_MILLISECONDS', 'TOTAL_FLOWS_EXP', 'TCP_FLAGS', 'PROTOCOL', 'TCP_WIN_MAX_OUT', 'SRC_TOS','ALERT' ]

'''

'''
 6 - Common features by overall weighted rank
'''

'''
req_cols =  [ 'FLOW_DURATION_MILLISECONDS', 'FIRST_SWITCHED', 'TOTAL_FLOWS_EXP', 'LAST_SWITCHED', 'TCP_WIN_SCALE_IN', 'TCP_WIN_MSS_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_MIN_IN', 'PROTOCOL', 'TCP_FLAGS','ALERT' ]

'''

'''
 7 - Common features by overall normalized weighted rank
'''

'''
req_cols =  [ 'TCP_WIN_SCALE_IN', 'FLOW_DURATION_MILLISECONDS', 'TCP_WIN_MSS_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_MIN_IN', 'TOTAL_FLOWS_EXP', 'FIRST_SWITCHED', 'TCP_FLAGS', 'TCP_WIN_MAX_OUT', 'PROTOCOL','ALERT' ]

'''

'''
 8 - Combined Selection
'''

'''

req_cols =  [ 'TCP_WIN_SCALE_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_MSS_IN', 'FLOW_DURATION_MILLISECONDS', 'TCP_WIN_MAX_OUT', 'TCP_FLAGS', 'PROTOCOL', 'TCP_WIN_MIN_OUT', 'TOTAL_FLOWS_EXP','ALERT']

'''


'''
##################################### For K = 5 ################################################
'''

'''
 1 - Common features by overall rank
'''

'''

req_cols =  [ 'TCP_WIN_SCALE_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_MSS_IN', 'TCP_FLAGS','ALERT' ]


'''

'''
 2 - Chi square
'''

'''

req_cols =  [ 'FLOW_DURATION_MILLISECONDS', 'PROTOCOL', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'TCP_WIN_MIN_IN','ALERT' ]


'''

'''
 3 - Feature Correlation
'''

'''
req_cols =  [ 'TCP_WIN_MSS_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_SCALE_IN', 'PROTOCOL','ALERT' ]

'''

'''
 4 - Feature Importance
'''

'''

req_cols =  [ 'TCP_WIN_MSS_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_SCALE_IN', 'TCP_FLAGS','ALERT' ]

'''

'''
 5 - Models + attacks
'''

'''
req_cols =  [ 'TCP_WIN_MSS_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_SCALE_IN', 'TCP_WIN_MIN_IN', 'FLOW_DURATION_MILLISECONDS', 'ALERT' ]

'''

'''
 6 - Common features by overall weighted rank
'''

'''
req_cols =  [ 'FLOW_DURATION_MILLISECONDS', 'FIRST_SWITCHED', 'TOTAL_FLOWS_EXP', 'LAST_SWITCHED', 'TCP_WIN_SCALE_IN','ALERT' ]

'''

'''
 7 - Common features by overall normalized weighted rank
'''

'''
req_cols =  [ 'TCP_WIN_SCALE_IN', 'FLOW_DURATION_MILLISECONDS', 'TCP_WIN_MSS_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_MIN_IN','ALERT' ]

'''

'''
 8 - Combined Selection
'''


req_cols =  [ 'TCP_WIN_SCALE_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_MSS_IN', 'FLOW_DURATION_MILLISECONDS','ALERT']





#----------------------------------------------------------#----------------------------------------------------------------
#Load Databases from csv file
print('---------------------------------------------------------------------------------')
print('loading databases')
print('---------------------------------------------------------------------------------')
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
#----------------------------------------------------------------
#----------------------------------------------------------------
frames = [df0, df1, df2, df3, df4, df5, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19]

df = pd.concat(frames,ignore_index=True)
#---------------------------------------------------------------------------------------------------#----------------------------------------------------------------
df = df.sample(frac =.1)
print('---------------------------------------------------------------------------------')
print('Normalizing Databases')
print('---------------------------------------------------------------------------------')

df_max_scaled = df.copy()
y = df_max_scaled.pop('ALERT')
df_max_scaled
for col in df_max_scaled.columns:
    t = abs(df_max_scaled[col].max())
#     print (t)
    df_max_scaled[col] = df_max_scaled[col]/t
df_max_scaled
df = df_max_scaled.assign(ALERT = y)
df = df.fillna(0)

#----------------------------------------------------------------#----------------------------------------------------------------


print('---------------------------------------------------------------------------------')
print('Spliting the db in training and testing')
print('---------------------------------------------------------------------------------')

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .70

train, test = df[df['is_train']==True], df[df['is_train']==False]
print('Number of the training data:', len(train))
print('Number of the testing data:', len(test))

features = df.columns[:26]

y_train, label = pd.factorize(train['ALERT'])
print('y_train: ',y_train)
y_test, label = pd.factorize(test['ALERT'])
print('y_test: ',y_test)


y = df.pop('ALERT')
X= df



X_train = train[features]
X_test = test[features]
#----------------------------------------------------------------#----------------------------------------------------------------

X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=0.70)
df = X.assign( ALERT = y)

#----------------------------------------
print('---------------------------------------------------------------------------------')
print('Balance Datasets')
print('---------------------------------------------------------------------------------')
print('')
counter = Counter(y_train)
print(counter)

# call balance operation until all labels have the same size
counter_list = list(counter.values())
for i in range(1,len(counter_list)):
    if counter_list[i-1] != counter_list[i]:
        X_train, y_train = oversample(X_train, y_train)

counter = Counter(y_train)
print('train len',counter)

# # After OverSampling training dataset

X_train = X_train.assign( ALERT = y_train)

X_train = X_train.sample(frac = 1)

#Drop ALert column from train
X_train.pop('ALERT')
labels_train_number, labels_train_label = pd.factorize(y_train)
labels_test_number, labels_test_label = pd.factorize(y_test)
# # Oversampling and balancing test data

counter = Counter(y_test)
print(counter)
counter_list = list(counter.values())
for i in range(1,len(counter_list)):
    if counter_list[i-1] != counter_list[i]:
        X_test, y_test = oversample(X_test,y_test)


counter = Counter(y_test)
print(counter)
counter_list = list(counter.values())
for i in range(1,len(counter_list)):
    if counter_list[i-1] != counter_list[i]:
        X_test, y_test = oversample(X_test,y_test)

counter = Counter(y_test)
print('test len ', counter)

#joining features and label
X_test = X_test.assign(ALERT = y_test)

#Randomize df order
X_test = X_test.sample(frac = 1)

#Drop label column
y_test = X_test.pop('ALERT')

y = y_test

# Separate features and labels
u, label = pd.factorize(y_train)
print('---------------------------------------------------------------------------------')
print('Defining MLP  Model')
print('---------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------')
print('Training Model')
print('---------------------------------------------------------------------------------')

start = time.time()
MLP = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
end = time.time()

print('---------------------------------------------------------------------------------')
print('ELAPSE TIME MODEL TRAINING: ',(end - start)/60, 'min')
print('---------------------------------------------------------------------------------')

start = time.time()
y_pred = MLP.predict_proba(X_test)
ynew = np.argmax(y_pred,axis = 1)

end = time.time()

print('---------------------------------------------------------------------------------')
print('ELAPSE TIME MODEL PREDICTION: ',(end - start)/60, 'min')
print('---------------------------------------------------------------------------------')

#----------------------------------------------------------------#----------------------------------------------------------------

print('---------------------------------------------------------------------------------')
print('CONFUSION MATRIX')
print('---------------------------------------------------------------------------------')

pred_label = label[ynew]
confusion_matrix = pd.crosstab(y_test, pred_label,rownames=['Actual ALERT'],colnames = ['Predicted ALERT'], dropna=False).sort_index(axis=0).sort_index(axis=1)
all_unique_values = sorted(set(pred_label) | set(y_test))
z = np.zeros((len(all_unique_values), len(all_unique_values)))
rows, cols = confusion_matrix.shape
z[:rows, :cols] = confusion_matrix
confusion_matrix  = pd.DataFrame(z, columns=all_unique_values, index=all_unique_values)
print(confusion_matrix)

FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.values.sum() - (FP + FN + TP)
TP_total = sum(TP)
TN_total = sum(TN)
FP_total = sum(FP)
FN_total = sum(FN)

TP_total = np.array(TP_total,dtype=np.float64)
TN_total = np.array(TN_total,dtype=np.float64)
FP_total = np.array(FP_total,dtype=np.float64)
FN_total = np.array(FN_total,dtype=np.float64)



#----------------------------------------------------------------#----------------------------------------------------------------

print('---------------------------------------------------------------------------------')
print('METRICS')
print('---------------------------------------------------------------------------------')

Acc = ACC(TP_total,TN_total, FP_total, FN_total)
Precision = PRECISION(TP_total, FP_total)
Recall = RECALL(TP_total, FN_total)
F1 = F1(Recall,Precision)
BACC = BACC(TP_total,TN_total, FP_total, FN_total)
MCC = MCC(TP_total,TN_total, FP_total, FN_total)
print('Accuracy total: ', Acc)
print('Precision total: ', Precision )
print('Recall total: ', Recall )
print('F1 total: ', F1 )
print('BACC total: ', BACC)
print('MCC total: ', MCC)
#----------------------------------------
for i in range(0,len(TP)):
    Acc = ACC(TP[i],TN[i], FP[i], FN[i])
    print('Accuracy: ', label[i] ,' - ' , Acc)
y_test_bin = label_binarize(y_test,classes = [0,1,2])
n_classes = y_test_bin.shape[1]
print('AUC_ROC total: ',AUC_ROC(y_test_bin,y_pred))
