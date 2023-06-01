# Let us Unveil Network Intrusion Features: Enhancing Network Intrusion Detection Systems via XAI-based Feature Selection

# Abstract 
The lack of performance evaluation and explainability of artificial intelligence (AI) models for intrusion detection systems (IDS) is a significant concern for human security analysts. In this context, feature selection is a crucial aspect of XAI where extracting the most significant features is essential for enhancing the explainability of results and assisting in the field of cyber security. In this study, we address such challenges of explaining AI for IDS where we propose novel methods for feature selection and create an explainable AI (XAI) framework for network intrusion detection.  We generate global explanations using SHapley Additive exPlanations (SHAP), extracting the most important features for all models and develop five novel feature selection methods from it. To evaluate our approach, we assess seven black-box AI models using two real-world network intrusion datasets. We establish a baseline without feature selection and gradually reduce the feature sets.  Additionally, we compare our SHAP-based methods with different state-of-the-art feature selection methods. Our framework offers innovative feature selection methods and provides a foundation for different XAI approaches which can help network security analysts in making more informed decisions. We openly share our source codes, offering a baseline XAI framework for the community to expand upon with new datasets and models. Our work contributes to the development of robust and interpretable AI models for network intrusion detection tasks.

# Performance 

Low-Level XAI Feature Selection Components

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/flow.png?raw=true)

Overall performances for AI models with different feature setups for the RoEduNet-SIMARGL2021 dataset.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/Ov1.png?raw=true)

Overall performances for AI models with different feature setups for the CICIDS-2017 dataset.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/Ov2.png?raw=true)

Accuracy per attack type (normal, DoS, and Port Scan) for the RoEduNet-SIMARGL2021 dataset.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/Acc1.png?raw=true)

Accuracy per attack type (normal, DoS, Brute Force, Web attack, Infiltration, Bot, and Port Scan) for the CICIDS-2017 dataset with all features.

![image](
https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/Acc2.png?raw=true)

Quantification of enhancements of AI models in detecting attacks (given by number of AI models with best performance) under feature selection.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/Qual.png?raw=true)

Comparison of AI performance under top features selected by our framework versus those by information gain and K-best. Our framework has superior performance (bold text) in 22 of 28 AI models for the two datasets.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/Comp.png?raw=true)



 
# How to use the programs:

## Performance with different feature selections.

- Download one of the datasets. RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee CICIDS-2017: https://www.kaggle.com/datasets/cicdataset/cicids2017
- The programs can be found inside the folder CICIDS-2017 or RoEduNet-SIMARGL2021 and end with sulfix acsac.
- Each program is a standalone program that is aimed to run one form of AI model within a set of features. (i.e. DNN_acsac.py in the CICIDS-2017 folder will run the DNN model with the selected features for that given dataset. Inside each program you can find a description of each feature selection method along with its features, the user has to uncomment the one to be used).
- Download that program 'utils.py' and leave it in the folder of the model program.
- Each program outputs a confusion matrix, metrics scores (i.e. accuracy (ACC), precision (Prec), recall (Rec), F1-score (F1), Matthews correlation coefficient (MCC), balanced accuracy (BACC), and the area under ROC curve (AUCROC)), and the accuracy per attack type.

## Different feature selection methods.

### Chi-square

- Download one of the datasets. RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee CICIDS-2017: https://www.kaggle.com/datasets/cicdataset/cicids2017
- The program can be found inside the folder CICIDS-2017 or RoEduNet-SIMARGL2021.
- Download that program 'utils.py' and leave it in the folder of the model program.
- The program is standalone program that outputs top features using the Chi-square method.

### Feature Correlation

- Download one of the datasets. RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee CICIDS-2017: https://www.kaggle.com/datasets/cicdataset/cicids2017
- The program can be found inside the folder CICIDS-2017 or RoEduNet-SIMARGL2021.
- Download that program 'utils.py' and leave it in the folder of the model program.
- The program is standalone program that outputs top features using the Feature Correlation method.

### Feature Importance

- Download one of the datasets. RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee CICIDS-2017: https://www.kaggle.com/datasets/cicdataset/cicids2017
- The program can be found inside the folder CICIDS-2017 or RoEduNet-SIMARGL2021.
- Download that program 'utils.py' and leave it in the folder of the model program.
- The program is standalone program that outputs top features using the Feature Importance method.

### Model Specific Features through SHAP (Used in methods below)

- Download one of the datasets. RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee CICIDS-2017: https://www.kaggle.com/datasets/cicdataset/cicids2017
- The programs can be found inside the folder CICIDS-2017/SHAP or RoEduNet-SIMARGL2021/SHAP and end with sulfix final.
- Each program is a standalone program that is aimed to run one form of AI model within a set of features. (i.e. DNN_final.py in the CICIDS-2017 folder will run the DNN model with 15 features for that given dataset. On the other hand. DNN_all_final.py will run the DNN model for all features for the given dataset).
- Download that program 'utils.py' and leave it in the folder of the model program.
- Each program outputs a confusion matrix, metrics scores (i.e. accuracy (ACC), precision (Prec), recall (Rec), F1-score (F1), Matthews correlation coefficient (MCC), balanced accuracy (BACC), and the area under ROC curve (AUCROC)), and the Global Summary/Beeswarm Plot.
- The most important features are extracted and shown in a list.

### Note: From here below Excel was used to compute feature selection
### Common features by overall rank

This method each feature rank for the models to create only one overall feature rank for all models. This is achieved by calculating the average rank of each individual feature across all AI models.

### Common features by overall weighted rank

This method builds upon the previous one. However, the difference is that it takes into consideration the SHAP values for each feature and the accuracy for each AI model, instead of the sequential numeration.
The importance of feature is calculated by the sum of the product of SHAP value and accuracy of that AI model for that feature. Then, the features are ranked according to average of that sum.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/assets/55901425/dec84f88-7d31-4c02-83b1-7f1d05e3394f)

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/assets/55901425/3863f8c2-5386-47db-9a14-10a089ff2b22)

### Common features by overall normalized weighted rank

This method is the same as the last one but with one difference. It normalizes all the SHAP values. During the experiments it was noted that some models such as LightGBM results in SHAP values with values that are much bigger others ones. Therefore, the normalization step was added to avoid such bias.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/assets/55901425/c2449822-9027-400e-9af0-0ddab9379296)

### Models + Attacks ranking score

This methods extracts significant intrusion features via selecting the top-𝑘 ranked features across all different AI models and all different intrusion types. Suppose the set of AI intrusion detection models is denoted by M in which each entry 𝑚 ∈ M represents one black-box AI model and that the set of intrusion types be given by A in which 𝑎 ∈ A represents one intrusion class. We calculate the overall ranking score of each feature (given by 𝑟𝑖) as follows:

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/assets/55901425/fff1bbb6-a561-4b98-ab2e-a9b984975785)

Where 𝑟𝑖𝑚 and 𝑟𝑖𝑎 are the ranks of feature 𝑖 for model 𝑚 ∈ M and intrusion 𝑎 ∈ A, respectively. The overall ranking score of a feature 𝑖 (𝑟𝑖) is given by the weighted sum of both the feature rank across all AI models and across all intrusion types. We then chose the 𝑘 features with lowest rank value. Note that the lower 𝑟𝑖, the higher the feature rank.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/assets/55901425/7ccc1944-3896-444c-9785-84d787e19dcf)

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/assets/55901425/4267a802-1344-487d-b4ce-a655f16674ca)

### Combined selection (Takes in consideration the seven methods used before)

In this method, we give a weight to each feature that depends on the frequency of appearance of this feature among top-𝑘 features in all proposed feature selection methods. In other words, the selection of feature here depends on its combined importance among all other proposed methods. We next show well known feature selection methods that are used in this work as baselines to our proposed methods. For these
methods, most of them do no need to train models beforehand.

