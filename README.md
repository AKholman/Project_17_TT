Data Projects (TripleTen)
 
 #Project_17

Project-17: Customer Churn Prediction for the Telecom Operator Interconnect

Project Overview
This project focuses on predicting customer churn for Interconnect, a telecom operator. The goal is to develop a predictive model to identify customers who are at risk of leaving, allowing the marketing team to offer promotions or special plans to retain them. By analyzing customer data, including personal information, service usage, and contract details, we aim to predict the likelihood of customer churn using several machine learning models.

Problem Statement
Churn prediction is crucial for telecom operators to reduce customer attrition and enhance customer retention strategies. In this project, we aim to predict whether a customer will leave the service (churn) by analyzing their demographic details, usage patterns, and contract information. By identifying at-risk customers, Interconnect can take preventive actions, such as offering discounts or targeted promotions.

Dataset
The dataset consists of several files, each containing different information related to customer behavior and service usage:

contract.csv — Contains details about the customer's contract, such as contract type, payment methods, and terms.
personal.csv — Contains personal information about customers like their demographic details.
internet.csv — Contains information about internet services the customer uses, such as whether they have DSL or fiber optic internet.
phone.csv — Contains data related to the customer's telephone services.
Each dataset includes the column customerID, which uniquely identifies each customer.

Target Feature:
EndDate column, where 'No' indicates a customer has churned, and 'Yes' means they are still a customer.
Evaluation Metrics:
Primary Metric: AUC-ROC score
Additional Metrics: Accuracy, F1 score
The target is to achieve an AUC-ROC score greater than 0.75 to ensure a good model performance.

Methodology
Step 1: Data Preprocessing
The project begins with preprocessing the data, cleaning and transforming the datasets into a format suitable for machine learning. This includes:

Handling missing values,
Feature engineering to define the features and target variables,
One-hot encoding for categorical features,
Standardization of numerical features to ensure that all features are on a similar scale.
Step 2: Exploratory Data Analysis (EDA)
EDA is performed to understand the underlying patterns in the data. This step includes:

Analyzing the distribution of customer demographics and contract details.
Visualizing correlations between features and churn.
Step 3: Model Development
Several machine learning models are trained and evaluated for predicting churn:

Logistic Regression
Random Forest Classifier
LightGBM Classifier
XGBoost
CatBoost
Each model is fine-tuned to optimize hyperparameters and evaluated using metrics such as Accuracy, ROC-AUC, and F1 score.

Step 4: Model Evaluation
The performance of each model is assessed, and the best model is selected based on the AUC-ROC score. The models are evaluated on a validation set first, and the best-performing model is then tested on the test set.

Results

|         Models	        |    Dataset	   |    Accuracy  |	  F1 Score  | 	ROC-AUC  |
|------------------------------------------------------------------------------------|
|  Logistic Regression	   |   Validation	 |    0.735	    |    0.595	  |   0.819    |
| Random Forest Classifier |	 Validation	 |    0.752	    |    0.593    | 	0.819    |
|  LightGBM	               |   Validation  |	  0.747	    |    0.599	  |   0.831    |
|  XGBoost	               |   Validation	 |    0.806     |	   0.565	  |   0.836    |
|  CatBoost                | 	 Validation	 |    0.793	    |    0.561	  |   0.842    |
|  CatBoost	               |   Test	       |    0.801	    |    0.573	  |   0.861    |
|------------------------------------------------------------------------------------|

After fine-tuning the models' hyperparameters, CatBoost emerged as the best model, achieving a ROC-AUC of 0.861 on the test set, making it the most reliable model for churn prediction.

Requirements
The following libraries are required to run the project:

  import pandas as pd
  import numpy as np
  from matplotlib import pyplot as plt
  from sklearn.model_selection import train_test_split
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
  from sklearn.pipeline import Pipeline
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score,f1_score, roc_auc_score, roc_curve, auc
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import GridSearchCV
  from lightgbm import LGBMClassifier
  from xgboost import XGBClassifier 
  from catboost import CatBoostClassifier 


Conclusion
Based on the results, CatBoost is recommended as the best model for predicting customer churn. It not only achieved a high ROC-AUC score on the test set but also provided better generalization compared to other models. By using the fine-tuned CatBoost model, Interconnect can effectively forecast which customers are at risk of churn and take proactive actions to retain them.

