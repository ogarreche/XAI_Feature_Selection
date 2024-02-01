##################################################
#               Parameter Setting                #
###################################################

fraction= 0.25 # how much of that database you want to use
frac_normal = .2 #how much of the normal classification you want to reduce
split = 0.70 # how you want to split the train/test data (this is percentage fro train)

#Model Parameters
max_iter=70

# XAI Samples
samples = 1#300


# Specify the name of the output text file
output_file_name = "MLP_LIME_CIC_output.txt"
with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)
###################################################
###################################################
###################################################

print('---------------------------------------------------------------------------------')
print('MLP')
print('---------------------------------------------------------------------------------')
print('Importing libraries ')
print('---------------------------------------------------------------------------------')

#----------------------------------------------------------------------
import tensorflow as tf
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
import keras
from keras.datasets import mnist
import shap
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
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
np.random.seed(0)
import sklearn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import lime
#----------------------------------------------------------------------
#---------------------------------------------------------------------
print('---------------------------------------------------------------------------------')
print('Defining Metrics')
print('---------------------------------------------------------------------------------')


def ACC(TP,TN,FP,FN):
    Acc = (TP+TN)/(TP+FP+FN+TN)
    return Acc
def ACC_2 (TP, FN):
    ac = (TP/(TP+FN))
    return ac
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
#-----------------------------------------------------------------------
#----------------------------------------------------------
print('---------------------------------------------------------------------------------')
print('Defining features')
print('---------------------------------------------------------------------------------')
'''
########################################### CICIDS Features ########################################
'''

# Select which feature method you want to use by uncommenting it.


'''
########################################### CICIDS Features ########################################
'''

# Select which feature method you want to use by uncommenting it.

'''
all features
'''

'''
req_cols = [' Destination Port',' Flow Duration',' Total Fwd Packets',' Total Backward Packets','Total Length of Fwd Packets',' Total Length of Bwd Packets',' Fwd Packet Length Max',' Fwd Packet Length Min',' Fwd Packet Length Mean',' Fwd Packet Length Std','Bwd Packet Length Max',' Bwd Packet Length Min',' Bwd Packet Length Mean',' Bwd Packet Length Std','Flow Bytes/s',' Flow Packets/s',' Flow IAT Mean',' Flow IAT Std',' Flow IAT Max',' Flow IAT Min','Fwd IAT Total',' Fwd IAT Mean',' Fwd IAT Std',' Fwd IAT Max',' Fwd IAT Min','Bwd IAT Total',' Bwd IAT Mean',' Bwd IAT Std',' Bwd IAT Max',' Bwd IAT Min','Fwd PSH Flags',' Bwd PSH Flags',' Fwd URG Flags',' Bwd URG Flags',' Fwd Header Length',' Bwd Header Length','Fwd Packets/s',' Bwd Packets/s',' Min Packet Length',' Max Packet Length',' Packet Length Mean',' Packet Length Std',' Packet Length Variance','FIN Flag Count',' SYN Flag Count',' RST Flag Count',' PSH Flag Count',' ACK Flag Count',' URG Flag Count',' CWE Flag Count',' ECE Flag Count',' Down/Up Ratio',' Average Packet Size',' Avg Fwd Segment Size',' Avg Bwd Segment Size',' Fwd Header Length','Fwd Avg Bytes/Bulk',' Fwd Avg Packets/Bulk',' Fwd Avg Bulk Rate',' Bwd Avg Bytes/Bulk',' Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets',' Subflow Fwd Bytes',' Subflow Bwd Packets',' Subflow Bwd Bytes','Init_Win_bytes_forward',' Init_Win_bytes_backward',' act_data_pkt_fwd',' min_seg_size_forward','Active Mean',' Active Std',' Active Max',' Active Min','Idle Mean',' Idle Std',' Idle Max',' Idle Min',' Label']
'''


'''
##################################### For K = 15 ################################################
'''

'''
 Information Gain according CICIDS paper
'''

# req_cols = [ ' Packet Length Std', ' Total Length of Bwd Packets', ' Subflow Bwd Bytes',
#     ' Destination Port', ' Packet Length Variance', ' Bwd Packet Length Mean',' Avg Bwd Segment Size',
#     'Bwd Packet Length Max', ' Init_Win_bytes_backward','Total Length of Fwd Packets',
#     ' Subflow Fwd Bytes', 'Init_Win_bytes_forward', ' Average Packet Size', ' Packet Length Mean',
#     ' Max Packet Length',' Label']

#  Our values SHAP
req_cols =  [ ' Bwd Packet Length Std', ' min_seg_size_forward', ' Average Packet Size', ' ACK Flag Count', ' Flow Duration', 'Bwd IAT Total', ' URG Flag Count', ' Avg Bwd Segment Size', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' PSH Flag Count', ' Destination Port',' Label' ]


'''
##################################### For K = 10 ################################################
'''
'''
 Information Gain according CICIDS paper
'''

# req_cols = [ ' Packet Length Std', ' Total Length of Bwd Packets', ' Subflow Bwd Bytes',
#     ' Destination Port', ' Packet Length Variance', ' Bwd Packet Length Mean',' Avg Bwd Segment Size',
#     'Bwd Packet Length Max', ' Init_Win_bytes_backward','Total Length of Fwd Packets',
#     ' Label']

'''
 1 - Common features by overall rank
'''

# req_cols =  [ ' Packet Length Std', ' Destination Port', 'Init_Win_bytes_forward', ' Packet Length Mean', ' Bwd Packet Length Mean', ' Average Packet Size', ' Init_Win_bytes_backward', ' Avg Bwd Segment Size', 'Bwd Packet Length Max', ' Packet Length Variance',' Label' ]


'''
 2 - Chi square
'''

'''
req_cols =  [ ' Destination Port', 'Bwd Packet Length Max', ' Bwd Packet Length Mean', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', ' Average Packet Size', ' Avg Bwd Segment Size', 'Init_Win_bytes_forward',' Label' ]
'''

'''
 3 - Feature Correlation
'''

'''
req_cols =  [ 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' Packet Length Variance', 'Bwd Packet Length Max', ' Packet Length Std', ' Destination Port', ' Bwd Packet Length Mean', ' Avg Bwd Segment Size', ' Max Packet Length', '  Subflow Fwd Bytes',' Label' ]
'''

'''
 4 - Feature Importance
'''

'''
req_cols =  [ ' Packet Length Std',  'Total Length of Bwd Packets', '  Subflow Bwd Bytes', ' Destination Port', ' Packet Length Variance', ' Bwd Packet Length Mean', ' Avg Bwd Segment Size', 'Bwd Packet Length Max', ' Init_Win_bytes_backward',  ' Total Length of Fwd Packets',' Label' ]
'''

'''
 5 - Models + attacks
'''

'''
req_cols =  [ ' Destination Port', ' Packet Length Std', 'Init_Win_bytes_forward', ' Avg Bwd Segment Size', ' Bwd Packet Length Mean', ' Packet Length Mean', ' Average Packet Size', ' Init_Win_bytes_backward', 'Bwd Packet Length Max', ' Max Packet Length',' Label' ]
'''

'''
 6 - Common features by overall weighted rank
'''

'''
req_cols =  [ ' Average Packet Size', ' Init_Win_bytes_backward', 'Init_Win_bytes_forward',  'Total Length of Fwd Packets', 'Bwd Packet Length Max', ' Packet Length Std', ' Packet Length Mean', ' Max Packet Length',  ' Total Length of Bwd Packets', ' Bwd Packet Length Mean',' Label' ]
'''

'''
 7 - Common features by overall normalized weighted rank
'''

'''
req_cols =  [ ' Average Packet Size', ' Destination Port', ' Packet Length Mean', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' Packet Length Std', ' Avg Bwd Segment Size', 'Bwd Packet Length Max', ' Bwd Packet Length Mean', ' Max Packet Length',' Label' ]
'''

'''
 8 - Combined Selection
'''

'''
req_cols =  [ ' Packet Length Std', 'Init_Win_bytes_forward', ' Bwd Packet Length Mean', 'Bwd Packet Length Max', ' Destination Port', ' Packet Length Mean', ' Average Packet Size', ' Init_Win_bytes_backward', ' Avg Bwd Segment Size', ' Max Packet Length',' Label' ]
'''


'''
##################################### For K = 5 ################################################
'''

'''
 1 - Common features by overall rank
'''

# req_cols =  [ ' Packet Length Std', ' Destination Port', 'Init_Win_bytes_forward', ' Packet Length Mean', ' Bwd Packet Length Mean', ' Label' ]


'''
 2 - Chi square
'''

'''
req_cols =  [ ' Destination Port', 'Bwd Packet Length Max', ' Bwd Packet Length Mean', ' Max Packet Length', ' Packet Length Mean', ' Label' ]
'''

'''
 3 - Feature Correlation
'''

'''
req_cols =  [ 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' Packet Length Variance', 'Bwd Packet Length Max', ' Packet Length Std', ' Label' ]
'''

'''
 4 - Feature Importance
'''

'''
req_cols =  [ ' Packet Length Std',  'Total Length of Bwd Packets', '  Subflow Bwd Bytes', ' Destination Port', ' Packet Length Variance', ' Label' ]
'''

'''
 5 - Models + attacks
'''

'''
req_cols =  [ ' Destination Port', ' Packet Length Std', 'Init_Win_bytes_forward', ' Avg Bwd Segment Size', ' Bwd Packet Length Mean', ' Label' ]
'''

'''
 6 - Common features by overall weighted rank
'''

'''
req_cols =  [ ' Average Packet Size', ' Init_Win_bytes_backward', 'Init_Win_bytes_forward',  'Total Length of Fwd Packets', 'Bwd Packet Length Max', ' Label' ]
'''

'''
 7 - Common features by overall normalized weighted rank
'''

'''
req_cols =  [ ' Average Packet Size', ' Destination Port', ' Packet Length Mean', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' Label' ]
'''

'''
 8 - Combined Selection
'''

# req_cols =  [ ' Packet Length Std', 'Init_Win_bytes_forward', ' Bwd Packet Length Mean', 'Bwd Packet Length Max', ' Destination Port', ' Label' ]



#Load Databases from csv file
print('---------------------------------------------------------------------------------')
print('loading databases')
print('---------------------------------------------------------------------------------')
df0 = pd.read_csv ('cicids_db/Wednesday-workingHours.pcap_ISCX.csv', usecols=req_cols)

df1 = pd.read_csv ('cicids_db/Tuesday-WorkingHours.pcap_ISCX.csv', usecols=req_cols)

df2 = pd.read_csv ('cicids_db/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', usecols=req_cols)

df3 = pd.read_csv ('cicids_db/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', usecols=req_cols)

df4 = pd.read_csv ('cicids_db/Monday-WorkingHours.pcap_ISCX.csv', usecols=req_cols)

df5 = pd.read_csv ('cicids_db/Friday-WorkingHours-Morning.pcap_ISCX.csv', usecols=req_cols)

df6 = pd.read_csv ('cicids_db/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', usecols=req_cols)

df7 = pd.read_csv ('cicids_db/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', usecols=req_cols)


frames = [df0, df1, df2, df3, df4, df5, df6, df7]

df = pd.concat(frames,ignore_index=True)

df = df.sample(frac =fraction)

y = df.pop(' Label')
df = df.assign(Label = y)
#---------------------------------------------------------------------------------------------------#----------------------------------------------------------------

print('---------------------------------------------------------------------------------')
print('Removing top features for acc')
print('---------------------------------------------------------------------------------')


# df.pop('FIN Flag Count')
# df.pop('Bwd Packet Length Max')
# df.pop('Idle Mean')
# df.pop('Bwd IAT Total')
# df.pop('Flow Bytes/s')
# df.pop(' ACK Flag Count')
# df.pop('Fwd Packets/s')
# df.pop('Fwd PSH Flags')
# df.pop(' Fwd Header Length')
# df.pop(' Fwd Avg Packets/Bulk')
# df.pop(' Fwd IAT Mean')
# df.pop(' Fwd Packet Length Max')
# df.pop('Init_Win_bytes_forward')
# df.pop(' Bwd PSH Flags')
# df.pop(' Fwd IAT Min')
# df.pop(' Active Std')
# df.pop(' min_seg_size_forward')
# df.pop(' Fwd IAT Std')
# df.pop('Fwd IAT Total')
# df.pop(' Fwd Packet Length Mean')
# df.pop(' Flow Packets/s')
# df.pop(' Bwd Packets/s')
# df.pop(' Total Fwd Packets')
# df.pop(' Fwd IAT Max')
# df.pop(' Flow IAT Std')
# df.pop('Subflow Fwd Packets')
# df.pop(' Idle Std')

# df.pop(' Fwd URG Flags')
# df.pop(' Fwd Packet Length Min')
# df.pop(' Bwd Packet Length Mean')
# df.pop(' CWE Flag Count')
# df.pop('Active Mean')
# df.pop(' Packet Length Variance')
# df.pop('Fwd Avg Bytes/Bulk')
# df.pop(' Bwd URG Flags')
# df.pop(' Total Backward Packets')
# df.pop(' Subflow Bwd Bytes')
# df.pop(' Fwd Avg Bulk Rate')
# df.pop(' act_data_pkt_fwd')
# df.pop(' Idle Max')
# df.pop(' Average Packet Size')
# df.pop(' Idle Min')
# df.pop('Bwd Avg Bulk Rate')
# df.pop(' SYN Flag Count')
# df.pop(' Fwd Packet Length Std')
# df.pop(' Flow IAT Mean')
# df.pop(' URG Flag Count')
# df.pop(' Active Min')
# df.pop(' Subflow Fwd Bytes')
# df.pop(' Min Packet Length')
# df.pop(' Bwd Packet Length Min')
# df.pop(' Init_Win_bytes_backward')
# df.pop(' Bwd IAT Min')
# df.pop(' Flow IAT Min')
# df.pop(' Subflow Bwd Packets')
# df.pop(' Packet Length Mean')
# df.pop(' Max Packet Length')
# df.pop(' PSH Flag Count')
# df.pop(' Flow IAT Max')
# df.pop(' RST Flag Count')
# df.pop(' Active Max')
# df.pop(' Packet Length Std')
# df.pop(' Avg Bwd Segment Size')
# df.pop(' Total Length of Bwd Packets')
# df.pop(' Bwd Header Length')
# df.pop(' Destination Port')
# df.pop(' ECE Flag Count')
# df.pop(' Bwd IAT Max')
# df.pop(' Bwd IAT Mean')
# df.pop(' Bwd Packet Length Std')
# df.pop(' Flow Duration')
# df.pop(' Down/Up Ratio')
# df.pop(' Avg Fwd Segment Size')
# df.pop(' Bwd Avg Packets/Bulk')
# df.pop(' Bwd IAT Std')
# df.pop(' Bwd Avg Bytes/Bulk')
print('---------------------------------------------------------------------------------')
print('Reducing Normal rows')
print('---------------------------------------------------------------------------------')
print('')


#filters

filtered_normal = df[df['Label'] == 'BENIGN']

#reduce

reduced_normal = filtered_normal.sample(frac=frac_normal)

#join

df = pd.concat([df[df['Label'] != 'BENIGN'], reduced_normal])

''' ---------------------------------------------------------------'''
df_max_scaled = df.copy()


y = df_max_scaled['Label'].replace({'DoS GoldenEye': 'Dos/Ddos', 'DoS Hulk': 'Dos/Ddos', 'DoS Slowhttptest': 'Dos/Ddos', 'DoS slowloris': 'Dos/Ddos', 'Heartbleed': 'Dos/Ddos', 'DDoS': 'Dos/Ddos','FTP-Patator': 'Brute Force', 'SSH-Patator': 'Brute Force','Web Attack - Brute Force': 'Web Attack', 'Web Attack - Sql Injection': 'Web Attack', 'Web Attack - XSS': 'Web Attack'})

df_max_scaled.pop('Label')



print('---------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------')
print('Normalizing')
print('---------------------------------------------------------------------------------')

for col in df_max_scaled.columns:
    t = abs(df_max_scaled[col].max())
    df_max_scaled[col] = df_max_scaled[col]/t
df_max_scaled
df = df_max_scaled.assign( Label = y)
df = df.fillna(0)

print('---------------------------------------------------------------------------------')
print('Balance Datasets')
print('---------------------------------------------------------------------------------')
print('')

counter = Counter(y)
print(counter)

# call balance operation until all labels have the same size
counter_list = list(counter.values())
for i in range(1,len(counter_list)):
    if counter_list[i-1] != counter_list[i]:
        df, y = oversample(df, y)

counter = Counter(y)


df = df.assign(Label = y)
print('train len',counter)

y = df.pop('Label')
X = df

df = df.assign(Label = y)


print('---------------------------------------------------------------------------------')
print('Spliting Train and Test')
print('---------------------------------------------------------------------------------')


X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=split)
df = X.assign( Label = y)

# Separate features and labels
u, label = pd.factorize(y_train)

#----------------------------------------------------------------#----------------------------------------------------------------

print('---------------------------------------------------------------------------------')
print('Defining MLP  Model')
print('---------------------------------------------------------------------------------')

print('---------------------------------------------------------------------------------')
print('Training Model')
print('---------------------------------------------------------------------------------')

start = time.time()

MLP = MLPClassifier(random_state=1, max_iter=max_iter).fit(X_train, y_train)

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
with open(output_file_name, "a") as f:print(confusion_matrix,file = f)

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
with open(output_file_name, "a") as f:print('Accuracy total: ', Acc, file = f)
print('Precision total: ', Precision )
print('Recall total: ', Recall )
print('F1 total: ', F1 )
print('BACC total: ', BACC)
print('MCC total: ', MCC)

# with open(output_file_name, "a") as f:print('Accuracy total: ', Acc, file = f)
with open(output_file_name, "a") as f:print('Precision total: ', Precision , file = f)
with open(output_file_name, "a") as f:print('Recall total: ', Recall,  file = f)
with open(output_file_name, "a") as f:print(    'F1 total: ', F1  , file = f)
with open(output_file_name, "a") as f:print(   'BACC total: ', BACC   , file = f)
with open(output_file_name, "a") as f:print(    'MCC total: ', MCC     , file = f)



for i in range(0,len(TP)):
   # Acc_2 = ACC_2(TP[i],FN[i])
    Acc = ACC(TP[i],TN[i], FP[i], FN[i])
    print('Accuracy: ', label[i] ,' - ' , Acc)
print('---------------------------------------------------------------------------------')



#----------------AUCROC--------------------
y = df.pop('Label')
X = df
y1, y2 = pd.factorize(y)

y_0 = pd.DataFrame(y1)
y_1 = pd.DataFrame(y1)
y_2 = pd.DataFrame(y1)
y_3 = pd.DataFrame(y1)
y_4 = pd.DataFrame(y1)
y_5 = pd.DataFrame(y1)
y_6 = pd.DataFrame(y1)

y_0 = y_0.replace(0, 0)
y_0 = y_0.replace(1, 1)
y_0 = y_0.replace(2, 1)
y_0 = y_0.replace(3, 1)
y_0 = y_0.replace(4, 1)
y_0 = y_0.replace(5, 1)
y_0 = y_0.replace(6, 1)

y_1 = y_1.replace(0, 1)
y_1 = y_1.replace(1, 0)
y_1 = y_1.replace(2, 1)
y_1 = y_1.replace(3, 1)
y_1 = y_1.replace(4, 1)
y_1 = y_1.replace(5, 1)
y_1 = y_1.replace(6, 1)

y_2 = y_2.replace(0, 1)
y_2 = y_2.replace(1, 1)
y_2 = y_2.replace(2, 0)
y_2 = y_2.replace(3, 1)
y_2 = y_2.replace(4, 1)
y_2 = y_2.replace(5, 1)
y_2 = y_2.replace(6, 1)

y_3 = y_3.replace(0, 1)
y_3 = y_3.replace(1, 1)
y_3 = y_3.replace(2, 1)
y_3 = y_3.replace(3, 0)
y_3 = y_3.replace(4, 1)
y_3 = y_3.replace(5, 1)
y_3 = y_3.replace(6, 1)

y_4 = y_4.replace(0, 1)
y_4 = y_4.replace(1, 1)
y_4 = y_4.replace(2, 1)
y_4 = y_4.replace(3, 1)
y_4 = y_4.replace(4, 0)
y_4 = y_4.replace(5, 1)
y_4 = y_4.replace(6, 1)

y_5 = y_5.replace(0, 1)
y_5 = y_5.replace(1, 1)
y_5 = y_5.replace(2, 1)
y_5 = y_5.replace(3, 1)
y_5 = y_5.replace(4, 1)
y_5 = y_5.replace(5, 0)
y_5 = y_5.replace(6, 1)

y_6 = y_6.replace(0, 1)
y_6 = y_6.replace(1, 1)
y_6 = y_6.replace(2, 1)
y_6 = y_6.replace(3, 1)
y_6 = y_6.replace(4, 1)
y_6 = y_6.replace(5, 1)
y_6 = y_6.replace(6, 0)

df = df.assign(Label = y)


#AUCROC
aucroc =[]
y_array = [y_0,y_1,y_2,y_3,y_4,y_5,y_6]
print('start AUCROC')
for j in range(0,len(y_array)):
    # print(j)
    #------------------------------------------------------------------------------------------------------------
    X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y_array[j], train_size=split)
    

    MLP = MLPClassifier(random_state=1, max_iter=max_iter).fit(X_train, y_train)
    y_pred = MLP.predict_proba(X_test)
    ynew = np.argmax(y_pred,axis = 1)

    y_scores = ynew
    y_true = y_test
    # Calculate AUC-ROC score
    auc_roc_score= roc_auc_score(y_true, y_scores,  average='weighted')  # Use 'micro' or 'macro' for different averaging strategies
    # print("AUC-ROC Score class:", auc_roc_score)
    aucroc.append(auc_roc_score)
    print(j,' - ok')
    #-------------------------------------------------------------------------------------------------------    -----
    # Calculate the average
average = sum(aucroc) / len(aucroc)

# Display the result
with open(output_file_name, "a") as f:print("AUC ROC Average:", average, file = f)
print("AUC ROC Average:", average)

#End AUC ROC






y_test_bin = label_binarize(y_test,classes = [0,1,2,3,4,5,6])
n_classes = y_test_bin.shape[1]
try:
    print('AUC_ROC total: ', roc_auc_score(y_test_bin,y_pred ,  multi_class='ovr'))
except:
    print('rocauc is nan')

with open(output_file_name, "a") as f:print('Accuracy total: ', Acc, file = f)
with open(output_file_name, "a") as f:print('Precision total: ', Precision , file = f)
with open(output_file_name, "a") as f:print('Recall total: ', Recall,  file = f)
with open(output_file_name, "a") as f:print(    'F1 total: ', F1  , file = f)
with open(output_file_name, "a") as f:print(   'BACC total: ', BACC   , file = f)
with open(output_file_name, "a") as f:print(    'MCC total: ', MCC     , file = f)
with open(output_file_name, "a") as f:print(   'AUC_ROC total: ', roc_auc_score(y_test_bin,y_pred ,  multi_class='ovr') , file = f)
