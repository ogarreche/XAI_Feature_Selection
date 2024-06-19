# Let us Unveil Network Intrusion Features: Enhancing Network Intrusion Detection Systems via XAI-based Feature Selection

# Abstract 
The lack of performance evaluation and explainability of artificial
intelligence (AI) models for intrusion detection systems (IDS) is a
significant concern for human security analysts. In this context,
feature selection is a crucial aspect of IDS where extracting the most
significant features is essential for enhancing the performance of
intrusion detection and understanding the main attributes that
identify anomalies in network security. In this study, we address
such challenge of feature selection for IDS where we propose
novel methods for feature selection using explainable AI (XAI)
methods. We extract the most important features of different AI
models and develop five novel feature selection methods from them.
To evaluate our approach, we assess seven black-box AI models
using two real-world network intrusion datasets. We establish
a baseline without feature selection and gradually reduce the
feature sets. We also compare our XAI-based feature selection
methods with different state-of-the-art feature selection methods
where we demonstrate that most AI models exhibit superior
performance when utilizing the top significant features selected
by our framework. Our work offers innovative feature selection
methods and provides a foundation for different XAI-based
approaches which can help network security analysts in making
more informed decisions based on knowledge of top intrusion
features. We release our source codes, offering a baseline XAI-based
feature selection framework for the community to expand upon
with new models. Our work provides an important contribution
towards the development of interpretable AI models for network
intrusion detection tasks by enhancing feature selection.

# Performance 

Low-Level XAI Feature Selection Components

![image](https://github.com/ogarreche/XAI_Feature_Selection/blob/main/images/flow.png?raw=true)

Overall performances for AI models with different feature setups for the RoEduNet-SIMARGL2021 dataset.

![image](https://github.com/ogarreche/XAI_Feature_Selection/blob/main/images/Ov1.png?raw=true)

Overall performances for AI models with different feature setups for the CICIDS-2017 dataset.

![image](https://github.com/ogarreche/XAI_Feature_Selection/blob/main/images/Ov2.png?raw=true)

Accuracy per attack type (normal, DoS, and Port Scan) for the RoEduNet-SIMARGL2021 dataset.

![image](https://github.com/ogarreche/XAI_Feature_Selection/blob/main/images/Acc1.png?raw=true)

Accuracy per attack type (normal, DoS, Brute Force, Web attack, Infiltration, Bot, and Port Scan) for the CICIDS-2017 dataset with all features.

![image](
https://github.com/ogarreche/XAI_Feature_Selection/blob/main/images/Acc2.png?raw=true)

Quantification of enhancements of AI models in detecting attacks (given by number of AI models with best performance) under feature selection.

![image](https://github.com/ogarreche/XAI_Feature_Selection/blob/main/images/Qual.png?raw=true)

Comparison of AI performance under top features selected by our framework versus those by information gain and K-best. Our framework has superior performance (bold text) in 22 of 28 AI models for the two datasets.

![image](https://github.com/ogarreche/XAI_Feature_Selection/blob/main/images/Comp.png?raw=true)



 
# How to use the programs:

## Performance with different feature selections.

- Download one of the datasets. RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee CICIDS-2017: https://www.kaggle.com/datasets/cicdataset/cicids2017
- The programs can be found inside the folder CICIDS-2017 or RoEduNet-SIMARGL2021 and end with sulfix final.
- Each program is a standalone program that is aimed to run one form of AI model within a set of features. (i.e. DNN_final.py in the CICIDS-2017 folder will run the DNN model with the selected features for that given dataset. Inside each program you can find a description of each feature selection method along with its features, the user has to uncomment the one to be used).
- Download that program 'utils.py' and leave it in the folder of the model program.
- Each program outputs a confusion matrix, metrics scores (i.e. accuracy (ACC), precision (Prec), recall (Rec), F1-score (F1), Matthews correlation coefficient (MCC), balanced accuracy (BACC), and the area under ROC curve (AUCROC)), and the accuracy per attack type.
- For Xplique, there is one jupyter notebook code for each dataset inside the xplique folder. Each program is standalone, and it ends with top features for each XAI method inside xplique. For more information about xplique see: https://github.com/deel-ai/xplique

## Different feature selection methods.

### Chi-square

- Download one of the datasets.
- 
RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee 

CICIDS-2017: [https://www.kaggle.com/datasets/cicdataset/cicids2017](https://www.kaggle.com/datasets/usmanshuaibumusa/cicids-17)

- The program can be found inside the folder CICIDS-2017 or RoEduNet-SIMARGL2021.
- Download that program 'utils.py' and leave it in the folder of the model program.
- The program is standalone program that outputs top features using the Chi-square method.

### Feature Correlation

- Download one of the datasets. 

RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee 

CICIDS-2017: [https://www.kaggle.com/datasets/cicdataset/cicids2017](https://www.kaggle.com/datasets/usmanshuaibumusa/cicids-17)


- The program can be found inside the folder CICIDS-2017 or RoEduNet-SIMARGL2021.
- Download that program 'utils.py' and leave it in the folder of the model program.
- The program is standalone program that outputs top features using the Feature Correlation method.

### Feature Importance

- Download one of the datasets. RoEduNet-SIMARGL2021:

RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee 

CICIDS-2017: [https://www.kaggle.com/datasets/cicdataset/cicids2017](https://www.kaggle.com/datasets/usmanshuaibumusa/cicids-17)



- The program can be found inside the folder CICIDS-2017 or RoEduNet-SIMARGL2021.
- Download that program 'utils.py' and leave it in the folder of the model program.
- The program is standalone program that outputs top features using the Feature Importance method.

### Model Specific Features through SHAP (Used in methods below)

- Download one of the datasets. 

RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee 

CICIDS-2017: [https://www.kaggle.com/datasets/cicdataset/cicids2017](https://www.kaggle.com/datasets/usmanshuaibumusa/cicids-17)


- The programs can be found inside the folder CICIDS-2017/SHAP or RoEduNet-SIMARGL2021/SHAP and end with sulfix final.
- Each program is a standalone program that is aimed to run one form of AI model within a set of features. (i.e. DNN_final.py in the CICIDS-2017 folder will run the DNN model with 15 features for that given dataset. On the other hand. DNN_all_final.py will run the DNN model for all features for the given dataset).
- Download that program 'utils.py' and leave it in the folder of the model program.
- Each program outputs a confusion matrix, metrics scores (i.e. accuracy (ACC), precision (Prec), recall (Rec), F1-score (F1), Matthews correlation coefficient (MCC), balanced accuracy (BACC), and the area under ROC curve (AUCROC)), and the Global Summary/Beeswarm Plot.
- The most important features are extracted and shown in a list.

### Note: From here below Excel was used to compute feature selection
### Common features by overall rank

This method each feature rank for the models to create only one overall feature rank for all models. This is achieved by calculating the average rank of each individual feature across all AI models.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/table1.png?raw=true)

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/table2.png?raw=true)

### Common features by overall weighted rank

This method builds upon the previous one. However, the difference is that it takes into consideration the SHAP values for each feature and the accuracy for each AI model, instead of the sequential numeration.
The importance of feature is calculated by the sum of the product of SHAP value and accuracy of that AI model for that feature. Then, the features are ranked according to average of that sum.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/table3.png?raw=true)

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/table4.png?raw=true)

### Common features by overall normalized weighted rank

This method is the same as the last one but with one difference. It normalizes all the SHAP values. During the experiments it was noted that some models such as LightGBM results in SHAP values with values that are much bigger others ones. Therefore, the normalization step was added to avoid such bias.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/table5.png?raw=true)

### Models + Attacks ranking score

This methods extracts significant intrusion features via selecting the top-𝑘 ranked features across all different AI models and all different intrusion types. Suppose the set of AI intrusion detection models is denoted by M in which each entry 𝑚 ∈ M represents one black-box AI model and that the set of intrusion types be given by A in which 𝑎 ∈ A represents one intrusion class. We calculate the overall ranking score of each feature (given by 𝑟𝑖) as follows:

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/form6.png?raw=true)

Where 𝑟𝑖𝑚 and 𝑟𝑖𝑎 are the ranks of feature 𝑖 for model 𝑚 ∈ M and intrusion 𝑎 ∈ A, respectively. The overall ranking score of a feature 𝑖 (𝑟𝑖) is given by the weighted sum of both the feature rank across all AI models and across all intrusion types. We then chose the 𝑘 features with lowest rank value. Note that the lower 𝑟𝑖, the higher the feature rank.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/table6.png?raw=true)

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/table8.png?raw=true)

### Combined selection (Takes in consideration the seven methods used before)

In this method, we give a weight to each feature that depends on the frequency of appearance of this feature among top-𝑘 features in all proposed feature selection methods. In other words, the selection of feature here depends on its combined importance among all other proposed methods. We next show well known feature selection methods that are used in this work as baselines to our proposed methods. For these
methods, most of them do no need to train models beforehand.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/table9.png?raw=true)

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/table10.png?raw=true)

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/table11.png?raw=true)

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/table12.png?raw=true)

### Example Program:

In the Example folder, first run both RF.ipynb and Light.ipynb to generate the SHAP graph and feature importance list. Then Check the Example.xlsx to generate the proposed methods (this process is manual). After completing this step, there will be new features list (common features by overall rank, etc) to be used again in the RF.ipynb and Light.ipynb to generate the metrics considering such features. 


