###################################################
#               Parameter Setting                #
###################################################

fraction= 0.01 # how much of that database you want to use
frac_normal = .2 #how much of the normal classification you want to reduce
split = 0.70 # how you want to split the train/test data (this is percentage fro train)

#Model Parameters
n_neighbors=5

# XAI Samples
samples =1 #50




output_file_name = "KNN_LIME_SML.txt"
with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)


###################################################
###################################################
###################################################

print('---------------------------------------------------------------------------------')
print('KNN')
print('---------------------------------------------------------------------------------')
print('Importing libraries ')
print('---------------------------------------------------------------------------------')

#----------------------------------------------------------------------
import tensorflow as tf
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from matplotlib import pyplot as plt
import numpy as np
# import pafy
import pandas as pd
#import csv
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
#from keras.utils import pad_sequences
from keras.preprocessing.sequence import TimeseriesGenerator
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
np.random.seed(0)
import shap

import sklearn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import lime



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
##################################### For K = 15 ################################################
'''

'''
 k gain according CICIDS paper
'''

'''
req_cols =  [ 'FLOW_DURATION_MILLISECONDS', 'FIRST_SWITCHED', 'TOTAL_FLOWS_EXP', 'TCP_WIN_MSS_IN', 'LAST_SWITCHED', 'TCP_WIN_MAX_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_MIN_OUT', 'PROTOCOL', 'TCP_WIN_MAX_OUT','ALERT' ]

'''


'''
 Our values SHAP
'''

req_cols =  [ 'PROTOCOL', 'FLOW_DURATION_MILLISECONDS', 'TCP_WIN_MAX_OUT', 'L4_SRC_PORT', 'TCP_WIN_MIN_OUT', 'FLOW_ID', 'TCP_FLAGS', 'IN_BYTES', 'SRC_TOS', 'TCP_WIN_SCALE_OUT', 'L4_DST_PORT', 'TCP_WIN_MAX_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_MSS_IN', 'TCP_WIN_SCALE_IN','ALERT' ]




'''
##################################### For K = 10 ################################################
'''

'''
 1 - Common features by overall rank
'''



# req_cols =  [ 'TCP_WIN_SCALE_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_MSS_IN', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS', 'TCP_WIN_MAX_OUT', 'TCP_WIN_MIN_OUT', 'SRC_TOS', 'DST_TOS','ALERT' ]




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


# req_cols =  [ 'TCP_WIN_SCALE_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_MSS_IN', 'TCP_FLAGS','ALERT' ]



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


'''
req_cols =  [ 'TCP_WIN_SCALE_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_MSS_IN', 'FLOW_DURATION_MILLISECONDS','ALERT']

'''





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

#Malware
# df6 = pd.read_csv ('sensor_db/malware-03-25-2022-17-57-07.csv', usecols=req_cols)

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
#frames = [df1,df13,df19]
df = pd.concat(frames,ignore_index=True)
#---------------------------------------------------------------------------------------------------#----------------------------------------------------------------
df = df.sample(frac =fraction)

# df.pop('TCP_WIN_SCALE_OUT')
# df.pop('TCP_WIN_MSS_IN')
# df.pop('TCP_WIN_MIN_IN')
# df.pop('TCP_WIN_MIN_OUT')
# df.pop('TCP_WIN_SCALE_IN')
# df.pop('TOTAL_BYTES_EXP')
# df.pop('TCP_WIN_MAX_IN')
# df.pop('FLOW_DURATION_MILLISECONDS')
# df.pop('TCP_WIN_MAX_OUT')
# df.pop('FLOW_ID')
# df.pop('DST_TOS')
# df.pop('IN_BYTES')
# df.pop('L4_SRC_PORT')
# df.pop('TCP_FLAGS')
# df.pop('FIRST_SWITCHED')
# df.pop('L4_DST_PORT')
# df.pop('IN_PKTS')
# df.pop('TOTAL_FLOWS_EXP')
# df.pop('MIN_IP_PKT_LEN')
# df.pop('MAX_IP_PKT_LEN')
# df.pop('LAST_SWITCHED')
# df.pop('OUT_BYTES')
# df.pop('SRC_TOS')
# df.pop('OUT_PKTS')
# df.pop('PROTOCOL')
# df.pop('TOTAL_PKTS_EXP')

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

#features = df.columns[:15]
features = df.columns[:len(req_cols)-1]

y_train, label = pd.factorize(train['ALERT'])
print('y_train: ',y_train)
y_test, label = pd.factorize(test['ALERT'])
print('y_test: ',y_test)


y = df.pop('ALERT')
X = df

X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=split)
df = X.assign( ALERT = y)



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

#----------------------------------------------------------------#----------------------------------------------------------------

print('---------------------------------------------------------------------------------')
print('Defining KNN Model')
print('---------------------------------------------------------------------------------')

knn_clf=KNeighborsClassifier(n_neighbors=n_neighbors)


print('---------------------------------------------------------------------------------')
print('Training Model')
print('---------------------------------------------------------------------------------')

start = time.time()
knn_clf.fit(X_train,y_train)
end = time.time()

print('---------------------------------------------------------------------------------')
print('ELAPSE TIME MODEL TRAINING: ',(end - start)/60, 'min')
print('---------------------------------------------------------------------------------')

start = time.time()
ypred=knn_clf.predict(X_test) #These are the predicted output value
end = time.time()

print('---------------------------------------------------------------------------------')
print('ELAPSE TIME MODEL PREDICTION: ',(end - start)/60, 'min')
print('---------------------------------------------------------------------------------')

y_pred = knn_clf.predict_proba(X_test)
#----------------------------------------------------------------#----------------------------------------------------------------

print('---------------------------------------------------------------------------------')
print('CONFUSION MATRIX')
print('---------------------------------------------------------------------------------')


pred_label = ypred
#pred_label = label[ypred]

confusion_matrix = pd.crosstab(y_test, pred_label,rownames=['Actual ALERT'],colnames = ['Predicted ALERT'], dropna=False).sort_index(axis=0).sort_index(axis=1)
all_unique_values = sorted(set(pred_label) | set(y_test))
z = np.zeros((len(all_unique_values), len(all_unique_values)))
rows, cols = confusion_matrix.shape
z[:rows, :cols] = confusion_matrix
confusion_matrix  = pd.DataFrame(z, columns=all_unique_values, index=all_unique_values)
confusion_matrix.to_csv('DNN_conf_matrix.csv')
with open(output_file_name, "a") as f:print(confusion_matrix,file=f)

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
with open(output_file_name, "a") as f:print('Accuracy total: ', Acc,file=f)
print('Precision total: ', Precision )
with open(output_file_name, "a") as f:print('Prec total: ', Precision,file=f)
print('Recall total: ', Recall )
with open(output_file_name, "a") as f:print('Rec total: ', Recall,file=f)
print('F1 total: ', F1 )
with open(output_file_name, "a") as f:print('f1 total: ', F1,file=f)
print('BACC total: ', BACC)
with open(output_file_name, "a") as f:print('BACC total: ', BACC,file=f)
print('MCC total: ', MCC)
with open(output_file_name, "a") as f:print('MCC total: ', MCC,file=f)
#----------------AUCROC--------------------
y = df.pop('ALERT')
X = df
y1, y2 = pd.factorize(y)

y_0 = pd.DataFrame(y1)
y_1 = pd.DataFrame(y1)
y_2 = pd.DataFrame(y1)

y_0 = y_0.replace(0, 0)
y_0 = y_0.replace(1, 1)
y_0 = y_0.replace(2, 1)


y_1 = y_1.replace(0, 1)
y_1 = y_1.replace(1, 0)
y_1 = y_1.replace(2, 1)


y_2 = y_2.replace(0, 1)
y_2 = y_2.replace(1, 1)
y_2 = y_2.replace(2, 0)



df = df.assign(Label = y)

#AUCROC
aucroc =[]
y_array = [y_0,y_1,y_2]
print('startAUCROC')
for j in range(0,len(y_array)):
    # print(j)
    #------------------------------------------------------------------------------------------------------------
    X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y_array[j], train_size=split)
    
    knn_clf=KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(X_train,y_train)
    y_pred=knn_clf.predict(X_test) #These are the predicted output value

    y_scores = y_pred
    y_true = y_test
    
    # Calculate AUC-ROC score
    auc_roc_score= roc_auc_score(y_true, y_scores,  average='weighted')  # Use 'micro' or 'macro' for different averaging strategies
    # print("AUC-ROC Score class:", auc_roc_score)
    aucroc.append(auc_roc_score)
    print('ok')
    #-------------------------------------------------------------------------------------------------------    -----
    # Calculate the average
average = sum(aucroc) / len(aucroc)

# Display the result
with open(output_file_name, "a") as f:print("AUC ROC Average:", average, file = f)
print("AUC ROC Average:", average)

#End AUC ROC



for i in range(0,len(TP)):
    Acc = ACC(TP[i],TN[i], FP[i], FN[i])
    print('Accuracy: ', label[i] ,' - ' , Acc)
#----------------------------------------
# y_test_bin = label_binarize(y_test,classes = [0,1,2])
# n_classes = y_test_bin.shape[1]
# print('AUC_ROC total: ',AUC_ROC(y_test_bin,y_pred))
#-----------------------------------------------------------------
