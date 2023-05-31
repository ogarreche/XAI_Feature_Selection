# Let us Unveil Network Intrusion Features: Enhancing Network Intrusion Detection Systems via XAI-based Feature Selection

# Abstract 
The lack of performance evaluation and explainability of artificial intelligence (AI) models for intrusion detection systems (IDS) is a significant concern for human security analysts. In this context, feature selection is a crucial aspect of XAI where extracting the most significant features is essential for enhancing the explainability of results and assisting in the field of cyber security. In this study, we address such challenges of explaining AI for IDS where we propose novel methods for feature selection and create an explainable AI (XAI) framework for network intrusion detection.  We generate global explanations using SHapley Additive exPlanations (SHAP), extracting the most important features for all models and develop five novel feature selection methods from it. To evaluate our approach, we assess seven black-box AI models using two real-world network intrusion datasets. We establish a baseline without feature selection and gradually reduce the feature sets.  Additionally, we compare our SHAP-based methods with different state-of-the-art feature selection methods. Our framework offers innovative feature selection methods and provides a foundation for different XAI approaches which can help network security analysts in making more informed decisions. We openly share our source codes, offering a baseline XAI framework for the community to expand upon with new datasets and models. Our work contributes to the development of robust and interpretable AI models for network intrusion detection tasks.

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)
![alt text](http://url/to/img.png)
# Performance 

Low-Level XAI Feature Selection Components

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/blob/main/images/flow.png?raw=true)

Overall performances for AI models with different feature setups for the RoEduNet-SIMARGL2021 dataset.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/assets/55901425/4a15e590-5851-40a9-9724-7e12ddf2f63d)

Overall performances for AI models with different feature setups for the CICIDS-2017 dataset.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/assets/55901425/4b48f04d-48ec-4973-b1cb-b9836847005e)

Accuracy per attack type (normal, DoS, and Port Scan) for the RoEduNet-SIMARGL2021 dataset.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/assets/55901425/7b365cea-ddbf-4030-a64c-a3d73cf7cb76)

Accuracy per attack type (normal, DoS, Brute Force, Web attack, Infiltration, Bot, and Port Scan) for the CICIDS-2017 dataset with all features.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/assets/55901425/70883e89-916c-4eea-850f-92d399922307)

Quantification of enhancements of AI models in detecting attacks (given by number of AI models with best performance) under feature selection.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/assets/55901425/6e434527-809c-485b-900d-9bf5e8cad178)

Comparison of AI performance under top features selected by our framework versus those by information gain and K-best. Our framework has superior performance (bold text) in 22 of 28 AI models for the two datasets.

![image](https://github.com/ogarreche/XAI_Feature_Selection_ACSAC_23/assets/55901425/25c06187-7c24-44fe-9936-49a83a728c7f)



 
# How to use the programs:

- Download one of the datasets. RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee CICIDS-2017: https://www.kaggle.com/datasets/cicdataset/cicids2017
- Each program is a standalone program that is aimed to run one form of AI model within a set of features. (i.e. DNN_acsac.py in the CICIDS-2017 folder will run the DNN model with the selected features for that given dataset. Inside each program you can find a description of each feature selection method along with its features, the user has to uncomment the one to be used).
Download that program 'utils.py' and leave it in the folder of the model program.
Each program outputs a confusion matrix, metrics scores (i.e. accuracy (ACC), precision (Prec), recall (Rec), F1-score (F1), Matthews correlation coefficient (MCC), balanced accuracy (BACC), and the area under ROC curve (AUCROC)), and the accuracy per attack type.
