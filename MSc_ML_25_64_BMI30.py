#!/usr/bin/env python
# coding: utf-8

# In[375]:


get_ipython().run_line_magic('matplotlib', 'inline')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from matplotlib import pyplot

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# In[376]:


df_2020 = pd.read_csv("C:/Users/Toomas/Desktop/ITMI_Data/Cleaned_datasets_25_64/Yearly/2020_cleaned_weights.csv")


# In[377]:


# drop irrelevant columns for analysis
df_2020 = df_2020.drop(columns=['survey_year', 'respondent_id', 'age','bmi','bmi_four_groups','bmi_two_groups_split25', 'weights'])


# In[378]:


# create dummy variables
# first change values from numeric to nominal for readable dummy labels

df_2020['gender'] = df_2020['gender'].replace([1,2],['MALE','FEMALE'])
df_2020['age_group'] = df_2020['age_group'].replace([1,2,3,4],['25_34','35_44','45_54','55_64'])
df_2020['ethnicity_estonian_nonestonian'] = df_2020['ethnicity_estonian_nonestonian'].replace([1,2],['ESTONIAN','NON_ESTONIAN'])
df_2020['education'] = df_2020['education'].replace([1,2,3,4],['PRIMARY_BASIC','SECONARY','SECONDARY_VOCATIONAL','HIGHER'])
df_2020['income_per_household_member'] = df_2020['income_per_household_member'].replace([1,2,3,4],['QUARTILE_1','QUARTILE_2','QUARTILE_3','QUARTILE_4'])
df_2020['chronic_disease'] = df_2020['chronic_disease'].replace([1,2],['YES','NO'])
df_2020['smoking_history'] = df_2020['smoking_history'].replace([1,2,3,4],['NEVER','FORMERLY','SELDOM','DAILY'])
df_2020['alcohol_standard_units_consumption_frequency'] = df_2020['alcohol_standard_units_consumption_frequency'].replace([1,2,3,4,5],['NEVER','<1x_MONTH','1+x_MONTH','1x_WEEK','ALMOST_DAILY'])
df_2020['exercising_frequency'] = df_2020['exercising_frequency'].replace([1,2,3,4,5,6],['NEVER','1x_MONTH','2_3x_MONTH','1x_WEEK','2_3x_WEEK', '4_7X_WEEK'])
df_2020['work_physical_effort_level'] = df_2020['work_physical_effort_level'].replace([1,2,3,4],['LITTLE','SOME','AVERAGE','A_LOT'])


df_dummies = pd.get_dummies(df_2020, columns=['gender', 'age_group',                                               'ethnicity_estonian_nonestonian', 'education',                                              'income_per_household_member', 'chronic_disease',                                              'smoking_history',                                              'alcohol_standard_units_consumption_frequency',                                              'exercising_frequency',                                              'work_physical_effort_level'],drop_first=False)


# In[379]:


df_dummies.columns


# In[380]:


## correlation graph
plt.style.use('ggplot')
plt.figure(figsize = (20,10))

# get a color map

my_cmap = cm.get_cmap('Accent')

# Get correlation of bmi_two_groups_split25 with other variables

df_dummies.corr()['bmi_two_groups_split30'].sort_values(ascending = False).plot(kind='bar', cmap=my_cmap)

# set titles for figure, x and 

plt.title('Correlation of bmi_two_groups_split30 with other features', fontsize=20)
plt.xlabel('Feature', fontsize=20)
plt.ylabel('BMI<30.0 or BMI≥30.0', fontsize=20)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/correlations.pdf")


# In[381]:


# scaling all variables to the range of 0 to 1 before data learning process

y = df_dummies['bmi_two_groups_split30'].values
X = df_dummies.drop(columns = ['bmi_two_groups_split30'])

features = X.columns.values
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features


# In[382]:


#split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

print("The shape of X_train dataset: ", X_train.shape)
print("The shape of y_train dataset: ", y_train.shape)
print("The shape of X_test dataset: ", X_test.shape)
print("The shape of y_test dataset: ", y_test.shape)


# In[383]:


# oversample the mintority class

sm = SMOTE(random_state=42)
X_train_res,y_train_res = sm.fit_resample(X_train,y_train)

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '2': {}".format(sum(y_train_res==2)))


# In[ ]:





# In[384]:


#========================================================
# 01A Logistic regression model training / original data
#=======================================================

model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)


# In[385]:


#print out the hyperparameters of the trained model
model_LR.get_params()


# In[386]:


# Logistic regression model prediction and metrics (original dataset)
y_pred = model_LR.predict(X_test)

print('LR model metrics:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: \t {0:.2f}'.format(metrics.precision_score(y_test, y_pred)))
print('Recall: \t {0:.2f}'.format(metrics.recall_score(y_test, y_pred)))
print('F1-score: \t {0:.2f}'.format(metrics.f1_score(y_test, y_pred)))


# In[387]:


# LR model performance with K-folds validation (original dataset)

from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model_LR_1 = LogisticRegression()
scores1 = cross_val_score(model_LR_1, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores2 = cross_val_score(model_LR_1, X, y, scoring='precision', cv=cv, n_jobs=-1)
scores3 = cross_val_score(model_LR_1, X, y, scoring='recall', cv=cv, n_jobs=-1)
scores4 = cross_val_score(model_LR_1, X, y, scoring='f1', cv=cv, n_jobs=-1)
print('Accuracy: %.2f' % (mean(scores1)))
print('Precision: %.2f' % (mean(scores2)))
print('Recall: %.2f' % (mean(scores3)))
print('F1 Score: %.2f' % (mean(scores4)))


# In[388]:


# get importance of features in LR model (original dataset)
importance = model_LR.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[389]:


# getting the weights of 10 most important features with association BMI≥30.0 in LR model (original dataset)

weights = pd.Series(model_LR.coef_[0], index = X.columns.values)
print(weights.sort_values(ascending=False)[:10].plot(kind='barh', color='grey'))
plt.tight_layout()
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/LR_features_most_BMI30.pdf", bbox_inches='tight')


# In[390]:


# getting the weights of 10 most important features with inverse association BMI≥30.0 in LR model (original dataset)
print(weights.sort_values(ascending=False)[-10:].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/LR_features_least_BMI30.pdf", bbox_inches='tight')


# In[391]:


# roc curve for LR model (original dataset)

pred_prob1 = model_LR.predict_proba(X_test)
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[392]:


# computing auc score for LR (original dataset)
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
print(auc_score1)


# In[393]:


# plot the roc curve for LR (original dataset)

plt.style.use('seaborn')


plt.plot(tpr1, fpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/ROC_curve_LR_BMI30.pdf")
plt.show();


# In[ ]:





# In[394]:


#====================================================
# 01B Logistic regression model training / oversampled data
#====================================================

model_LR_res = LogisticRegression()
model_LR_res.fit(X_train_res, y_train_res)


# In[395]:



# Logistic regression model prediction and metrics (oversampled data)
y_pred = model_LR_res.predict(X_test)

print('LR model metrics:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: \t {0:.2f}'.format(metrics.precision_score(y_test, y_pred)))
print('Recall: \t {0:.2f}'.format(metrics.recall_score(y_test, y_pred)))
print('F1-score: \t {0:.2f}'.format(metrics.f1_score(y_test, y_pred)))


# In[396]:


# LR model performance with K-folds validation (oversampled data)

from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#oversamplling the whole dataset (not just training data)
sm_all = SMOTE(random_state=42)
X_res,y_res = sm_all.fit_resample(X,y)

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model_LR_1_res = LogisticRegression()
scores1 = cross_val_score(model_LR_1_res, X_res, y_res, scoring='accuracy', cv=cv, n_jobs=-1)
scores2 = cross_val_score(model_LR_1_res, X_res, y_res, scoring='precision', cv=cv, n_jobs=-1)
scores3 = cross_val_score(model_LR_1_res, X_res, y_res, scoring='recall', cv=cv, n_jobs=-1)
scores4 = cross_val_score(model_LR_1_res, X_res, y_res, scoring='f1', cv=cv, n_jobs=-1)
print('Accuracy: %.2f' % (mean(scores1)))
print('Precision: %.2f' % (mean(scores2)))
print('Recall: %.2f' % (mean(scores3)))
print('F1 Score: %.2f' % (mean(scores4)))


# In[397]:


# get importance of features in LR model (oversampled dataset)
importance = model_LR_res.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[398]:


# getting the weights of 10 most important features with association BMI≥30.0 in LR model (oversampled data)

weights = pd.Series(model_LR_res.coef_[0], index = X.columns.values)
print(weights.sort_values(ascending=False)[:10].plot(kind='barh', color='grey'))
plt.tight_layout()
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/LR_features_most_BMI30_res.pdf", bbox_inches='tight')


# In[399]:


# getting the weights of 10 most important features with inverse association BMI≥30.0 in LR model (oversampled dataset)
print(weights.sort_values(ascending=False)[-10:].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/LR_features_least_BMI30_res.pdf", bbox_inches='tight')


# In[400]:


# roc curve for LR model (oversampled data)

pred_prob1 = model_LR_res.predict_proba(X_test)
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[401]:


# computing auc score for LR (oversampled data)
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
print(auc_score1)


# In[402]:


# plot the roc curve for LR (oversampled data)

plt.style.use('seaborn')


plt.plot(tpr1, fpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/ROC_curve_LR_BMI30_res.pdf")
plt.show();


# In[ ]:





# In[403]:


#====================================
# 02A Random Forest model training
#====================================

model_RF = RandomForestClassifier()

model_RF.fit(X_train, y_train)


# In[404]:


#print out the hyperparameters of the trained model (original data)
model_RF.get_params()


# In[405]:


# Random forest model prediction and metrics (original data)

y_pred = model_RF.predict(X_test)

print('RF model metrics:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: \t {0:.2f}'.format(metrics.precision_score(y_test, y_pred)))
print('Recall: \t {0:.2f}'.format(metrics.recall_score(y_test, y_pred)))
print('F1-score: \t {0:.2f}'.format(metrics.f1_score(y_test, y_pred)))


# In[406]:


# RF model performance with K-folds validation (original data)

from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model_RF_1 = RandomForestClassifier()
scores1 = cross_val_score(model_RF_1, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores2 = cross_val_score(model_RF_1, X, y, scoring='precision', cv=cv, n_jobs=-1)
scores3 = cross_val_score(model_RF_1, X, y, scoring='recall', cv=cv, n_jobs=-1)
scores4 = cross_val_score(model_RF_1, X, y, scoring='f1', cv=cv, n_jobs=-1)
print('Accuracy: %.2f' % (mean(scores1)))
print('Precision: %.2f' % (mean(scores2)))
print('Recall: %.2f' % (mean(scores3)))
print('F1 Score: %.2f' % (mean(scores4)))


# In[407]:


# calculation of RF model feature importance (original data)

importance = model_RF.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[408]:


# 10 most important features/ BMI≥30.0 in RF model (original data)

weights = pd.Series(model_RF.feature_importances_, index = X.columns.values)
print(weights.sort_values(ascending=False)[:10].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/RF_features_most_BMI30.pdf", bbox_inches='tight')


# In[409]:


# 10 least important features/ BMI≥30.0 in RF model (original data)
print(weights.sort_values(ascending=False)[-10:].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/RF_features_least_BMI30.pdf", bbox_inches='tight')


# In[410]:


# roc curve for RF model (original data)

pred_prob2 = model_RF.predict_proba(X_test)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[411]:


# computing auc score for RF model (original data)
auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])
print(auc_score2)


# In[412]:


# plot roc curve for RF model (original data)

plt.style.use('seaborn')


plt.plot(tpr2, fpr2, linestyle='--',color='green', label='Random Forest')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/ROC_curve_RF_BMI30.pdf")
plt.show();


# In[ ]:





# In[413]:


#===================================================
# 02B Random Forest model training (oversampled data)
#===================================================

model_RF_res = RandomForestClassifier()
model_RF_res.fit(X_train_res, y_train_res)


# In[414]:


# Random forest model prediction and metrics (oversampled data)

y_pred = model_RF_res.predict(X_test)

print('RF model metrics:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: \t {0:.2f}'.format(metrics.precision_score(y_test, y_pred)))
print('Recall: \t {0:.2f}'.format(metrics.recall_score(y_test, y_pred)))
print('F1-score: \t {0:.2f}'.format(metrics.f1_score(y_test, y_pred)))


# In[415]:


# RF model performance with K-folds validation (oversampled data)

from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#oversamplling the whole dataset (not just training data)
sm_all = SMOTE(random_state=42)
X_res,y_res = sm_all.fit_resample(X,y)

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model_RF_1_res = RandomForestClassifier()
scores1 = cross_val_score(model_RF_1_res, X_res, y_res, scoring='accuracy', cv=cv, n_jobs=-1)
scores2 = cross_val_score(model_RF_1_res, X_res, y_res, scoring='precision', cv=cv, n_jobs=-1)
scores3 = cross_val_score(model_RF_1_res, X_res, y_res, scoring='recall', cv=cv, n_jobs=-1)
scores4 = cross_val_score(model_RF_1_res, X_res, y_res, scoring='f1', cv=cv, n_jobs=-1)
print('Accuracy: %.2f' % (mean(scores1)))
print('Precision: %.2f' % (mean(scores2)))
print('Recall: %.2f' % (mean(scores3)))
print('F1 Score: %.2f' % (mean(scores4)))


# In[416]:


# calculation of RF model feature importance (oversampled data)

importance = model_RF_res.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[417]:


# 10 most important features/ BMI≥30.0 in RF model (oversampled data)

weights = pd.Series(model_RF_res.feature_importances_, index = X.columns.values)
print(weights.sort_values(ascending=False)[:10].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/RF_features_most_BMI30_res.pdf", bbox_inches='tight')


# In[418]:


# 10 least important features/ BMI≥30.0 in RF model (oversampled data)
print(weights.sort_values(ascending=False)[-10:].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/RF_features_least_BMI30_res.pdf", bbox_inches='tight')


# In[419]:


# roc curve for RF model (oversampled data)

pred_prob2 = model_RF_res.predict_proba(X_test)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[420]:


# computing auc score for RF model (oversampled data)
auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])
print(auc_score2)


# In[421]:


# plot roc curve for RF model (oversampled data)

plt.style.use('seaborn')


plt.plot(tpr2, fpr2, linestyle='--',color='green', label='Random Forest')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/ROC_curve_RF_BMI30_res.pdf")
plt.show();


# In[ ]:





# In[422]:


#====================================
# 03A SVM model training (original data)
#====================================

model_SVM=SVC(probability = True, kernel ='linear')
model_SVM.fit(X_train, y_train)


# In[423]:


#print out the hyperparameters of the trained model (original data)
model_SVM.get_params()


# In[424]:


# SVM model prediction and metrics (original data)

y_pred = model_SVM.predict(X_test)

print('SVM model metrics:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: \t {0:.2f}'.format(metrics.precision_score(y_test, y_pred)))
print('Recall: \t {0:.2f}'.format(metrics.recall_score(y_test, y_pred)))
print('F1-score: \t {0:.2f}'.format(metrics.f1_score(y_test, y_pred)))


# In[425]:


# SVM model performance with K-folds validation (original data)

from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model_SVM_1 = SVC(probability = True, kernel ='linear')
scores1 = cross_val_score(model_SVM_1, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores2 = cross_val_score(model_SVM_1, X, y, scoring='precision', cv=cv, n_jobs=-1)
scores3 = cross_val_score(model_SVM_1, X, y, scoring='recall', cv=cv, n_jobs=-1)
scores4 = cross_val_score(model_SVM_1, X, y, scoring='f1', cv=cv, n_jobs=-1)
print('Accuracy: %.2f' % (mean(scores1)))
print('Precision: %.2f' % (mean(scores2)))
print('Recall: %.2f' % (mean(scores3)))
print('F1 Score: %.2f' % (mean(scores4)))


# In[426]:


# 10 most important variables affecting BMI≥30.0 in SVM model (original data) 

weights = pd.Series(model_SVM.coef_[0], index = X.columns.values)
print(weights.sort_values(ascending=False)[:10].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/SVM_features_most_BMI30.pdf", bbox_inches='tight')


# In[427]:


# # 10 least important variables affecting BMI≥30.0 in SVM model (original data) 
print(weights.sort_values(ascending=False)[-10:].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/SVM_features_least_BMI30.pdf", bbox_inches='tight')


# In[428]:


# roc curve for SVM model (original data) 

pred_prob3 = model_SVM.predict_proba(X_test)
fpr3, tpr3, thresh3 = roc_curve(y_test, pred_prob3[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[429]:


# computing auc score for SVM (original data) 
auc_score3 = roc_auc_score(y_test, pred_prob3[:,1])
print(auc_score3)


# In[430]:


# plot roc curve for SVM model (original data)
plt.style.use('seaborn')


plt.plot(tpr3, fpr3, linestyle='--',color='violet', label='SVM')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/ROC_curve_SVM_BMI30.pdf")
plt.show();


# In[ ]:





# In[431]:


#==========================================
# 03B SVM model training (oversampled data)
#==========================================

model_SVM_res=SVC(probability = True, kernel ='linear')
model_SVM_res.fit(X_train_res, y_train_res)


# In[432]:


# SVM model prediction and metrics (oversampled data)

y_pred = model_SVM_res.predict(X_test)

print('SVM model metrics:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: \t {0:.2f}'.format(metrics.precision_score(y_test, y_pred)))
print('Recall: \t {0:.2f}'.format(metrics.recall_score(y_test, y_pred)))
print('F1-score: \t {0:.2f}'.format(metrics.f1_score(y_test, y_pred)))


# In[433]:


# RF model performance with K-folds validation (oversampled data)


#oversamplling the whole dataset (not just training data)
sm_all = SMOTE(random_state=1)
X_res,y_res = sm_all.fit_resample(X,y)

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model_SVM_1_res = SVC(probability = True, kernel ='linear')
scores1 = cross_val_score(model_SVM_1_res, X_res, y_res, scoring='accuracy', cv=cv, n_jobs=-1)
scores2 = cross_val_score(model_SVM_1_res, X_res, y_res, scoring='precision', cv=cv, n_jobs=-1)
scores3 = cross_val_score(model_SVM_1_res, X_res, y_res, scoring='recall', cv=cv, n_jobs=-1)
scores4 = cross_val_score(model_SVM_1_res, X_res, y_res, scoring='f1', cv=cv, n_jobs=-1)
print('Accuracy: %.2f' % (mean(scores1)))
print('Precision: %.2f' % (mean(scores2)))
print('Recall: %.2f' % (mean(scores3)))
print('F1 Score: %.2f' % (mean(scores4)))


# In[434]:


# 10 most important variables affecting BMI≥30.0 in SVM model (oversampled data) 

weights = pd.Series(model_SVM_res.coef_[0], index = X.columns.values)
print(weights.sort_values(ascending=False)[:10].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/SVM_features_most_BMI30_res.pdf", bbox_inches='tight')


# In[435]:


# # 10 least important variables affecting BMI≥30.0 in SVM model (oversampled data) 
print(weights.sort_values(ascending=False)[-10:].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/SVM_features_least_BMI30_res.pdf", bbox_inches='tight')


# In[436]:


# roc curve for SVM model (oversampled data) 

pred_prob3 = model_SVM_res.predict_proba(X_test)
fpr3, tpr3, thresh3 = roc_curve(y_test, pred_prob3[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[437]:


# computing auc score for SVM (oversampled data) 
auc_score3 = roc_auc_score(y_test, pred_prob3[:,1])
print(auc_score3)


# In[438]:


# plot roc curve for SVM model (oversampled data)
plt.style.use('seaborn')


plt.plot(tpr3, fpr3, linestyle='--',color='violet', label='SVM')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/ROC_curve_SVM_BMI30_res.pdf")
plt.show();


# In[ ]:





# In[439]:


#=============================================
# Decision Tree model training (original data)
#=============================================

model_DT = DecisionTreeClassifier()
model_DT.fit(X_train, y_train)


# In[440]:


#print out the hyperparameters of the trained model (original data)
model_DT.get_params()


# In[441]:


# DT model prediction and metrics (original data)

y_pred = model_DT.predict(X_test)

print('DT model metrics:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: \t {0:.2f}'.format(metrics.precision_score(y_test, y_pred)))
print('Recall: \t {0:.2f}'.format(metrics.recall_score(y_test, y_pred)))
print('F1-score: \t {0:.2f}'.format(metrics.f1_score(y_test, y_pred)))


# In[442]:


# DT model performance with K-folds validation (original data)

from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model_DT_1 = DecisionTreeClassifier()
scores1 = cross_val_score(model_DT_1, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores2 = cross_val_score(model_DT_1, X, y, scoring='precision', cv=cv, n_jobs=-1)
scores3 = cross_val_score(model_DT_1, X, y, scoring='recall', cv=cv, n_jobs=-1)
scores4 = cross_val_score(model_DT_1, X, y, scoring='f1', cv=cv, n_jobs=-1)
print('Accuracy: %.2f' % (mean(scores1)))
print('Precision: %.2f' % (mean(scores2)))
print('Recall: %.2f' % (mean(scores3)))
print('F1 Score: %.2f' % (mean(scores4)))


# In[443]:


# plot feature important of DT/CART model (original data)
importance = model_DT.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[444]:


# 10 most important variables affecting BMI≥25.0 in DT model  (original data)

weights = pd.Series(model_DT.feature_importances_, index = X.columns.values)
print(weights.sort_values(ascending=False)[:10].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/DT_features_most_BMI30.pdf", bbox_inches='tight')


# In[445]:


# 10 least important variables affecting BMI≥25.0 in DT model  (original data)
print(weights.sort_values(ascending=False)[-10:].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/DT_features_least_BMI30.pdf", bbox_inches='tight')


# In[446]:


# roc curve for DT model   (original data)

pred_prob4 = model_DT.predict_proba(X_test)
fpr4, tpr4, thresh4 = roc_curve(y_test, pred_prob4[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[447]:


# computing auc score for DT   (original data)
auc_score4 = roc_auc_score(y_test, pred_prob4[:,1])
print(auc_score4)


# In[448]:


# plot roc curve for DT  (original data)

plt.style.use('seaborn')


plt.plot(tpr4, fpr4, linestyle='--',color='black', label='DT')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/ROC_curve_DT_BMI30.pdf")
plt.show();


# In[ ]:





# In[449]:


#=============================================
# Decision Tree model training (oversampled data)
#=============================================

model_DT_res = DecisionTreeClassifier()
model_DT_res.fit(X_train_res, y_train_res)


# In[450]:


# DT model prediction and metrics (oversampled data)

y_pred = model_DT_res.predict(X_test)

print('DT model metrics:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: \t {0:.2f}'.format(metrics.precision_score(y_test, y_pred)))
print('Recall: \t {0:.2f}'.format(metrics.recall_score(y_test, y_pred)))
print('F1-score: \t {0:.2f}'.format(metrics.f1_score(y_test, y_pred)))


# In[451]:


# DT model performance with K-folds validation (oversampled data)


#oversamplling the whole dataset (not just training data)
sm_all = SMOTE(random_state=42)
X_res,y_res = sm_all.fit_resample(X,y)

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model_DT_1_res = DecisionTreeClassifier()
scores1 = cross_val_score(model_DT_1_res, X_res, y_res, scoring='accuracy', cv=cv, n_jobs=-1)
scores2 = cross_val_score(model_DT_1_res, X_res, y_res, scoring='precision', cv=cv, n_jobs=-1)
scores3 = cross_val_score(model_DT_1_res, X_res, y_res, scoring='recall', cv=cv, n_jobs=-1)
scores4 = cross_val_score(model_DT_1_res, X_res, y_res, scoring='f1', cv=cv, n_jobs=-1)
print('Accuracy: %.2f' % (mean(scores1)))
print('Precision: %.2f' % (mean(scores2)))
print('Recall: %.2f' % (mean(scores3)))
print('F1 Score: %.2f' % (mean(scores4)))


# In[452]:


# 10 most important variables affecting BMI≥25.0 in DT model  (oversampled data)

weights = pd.Series(model_DT_res.feature_importances_, index = X.columns.values)
print(weights.sort_values(ascending=False)[:10].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/DT_features_most_BMI30_RES.pdf", bbox_inches='tight')


# In[453]:


# 10 least important variables affecting BMI≥25.0 in DT model  (oversampled data)
print(weights.sort_values(ascending=False)[-10:].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/DT_features_least_BMI30.pdf", bbox_inches='tight')


# In[454]:


# roc curve for DT model   (oversampled data)

pred_prob4 = model_DT_res.predict_proba(X_test)
fpr4, tpr4, thresh4 = roc_curve(y_test, pred_prob4[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[455]:


auc_score4 = roc_auc_score(y_test, pred_prob4[:,1])
print(auc_score4)


# In[456]:


# plot roc curve for DT  (oversampled data)

plt.style.use('seaborn')


plt.plot(tpr4, fpr4, linestyle='--',color='black', label='DT')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/ROC_curve_DT_BMI30_res.pdf")
plt.show();


# In[ ]:





# In[457]:


#====================================================
# Gaussian Naive Bayes model training (original data)
#====================================================

model_NB = GaussianNB()
model_NB.fit(X_train, y_train)


# In[458]:


#print out the hyperparameters of the trained model (original data)
model_NB.get_params()


# In[459]:


# NB model prediction and metrics (original data)

y_pred = model_NB.predict(X_test)

print('NB model metrics:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: \t {0:.2f}'.format(metrics.precision_score(y_test, y_pred)))
print('Recall: \t {0:.2f}'.format(metrics.recall_score(y_test, y_pred)))
print('F1-score: \t {0:.2f}'.format(metrics.f1_score(y_test, y_pred)))


# In[460]:


# DT model performance with K-folds validation (original data)

from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model_NB_1 = GaussianNB()
scores1 = cross_val_score(model_NB_1, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores2 = cross_val_score(model_NB_1, X, y, scoring='precision', cv=cv, n_jobs=-1)
scores3 = cross_val_score(model_NB_1, X, y, scoring='recall', cv=cv, n_jobs=-1)
scores4 = cross_val_score(model_NB_1, X, y, scoring='f1', cv=cv, n_jobs=-1)
print('Accuracy: %.2f' % (mean(scores1)))
print('Precision: %.2f' % (mean(scores2)))
print('Recall: %.2f' % (mean(scores3)))
print('F1 Score: %.2f' % (mean(scores4)))


# In[461]:


# plot feature important of NB model (original data)

from sklearn.inspection import permutation_importance

imps = permutation_importance(model_NB, X_test, y_test)
importances = imps.importances_mean
std = imps.importances_std
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_test.shape[1]):
    print("%d. %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))


# In[462]:


# greatest impact / NB model  (original data)

weights = pd.Series(importances, index = X.columns.values)
print(weights.sort_values(ascending=False)[:10].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/NB_features_most_BMI30.pdf", bbox_inches='tight')


# In[463]:


# least impact / NB model  (original data)
print(weights.sort_values(ascending=False)[-10:].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/NB_features_least_BMI30.pdf", bbox_inches='tight')


# In[464]:


# roc curve for NB model  (original data)

pred_prob5 = model_NB.predict_proba(X_test)
fpr5, tpr5, thresh5 = roc_curve(y_test, pred_prob5[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[465]:


# computing auc score for NB  (original data)
auc_score5 = roc_auc_score(y_test, pred_prob5[:,1])
print(auc_score5)


# In[466]:


# plot roc curves for NB  (original data)

plt.style.use('seaborn')

plt.plot(tpr5, fpr5, linestyle='--',color='red', label='NB')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/ROC_curve_NB_BMI30.pdf")
plt.show();


# In[ ]:





# In[467]:


#====================================================
# Gaussian Naive Bayes model training (oversampled data)
#====================================================

model_NB_res = GaussianNB()
model_NB_res.fit(X_train_res, y_train_res)


# In[468]:


# NB model prediction and metrics (oversampled data)

y_pred = model_NB_res.predict(X_test)

print('NB model metrics:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: \t {0:.2f}'.format(metrics.precision_score(y_test, y_pred)))
print('Recall: \t {0:.2f}'.format(metrics.recall_score(y_test, y_pred)))
print('F1-score: \t {0:.2f}'.format(metrics.f1_score(y_test, y_pred)))


# In[469]:


# DT model performance with K-folds validation (oversampled data)


#oversamplling the whole dataset (not just training data)
sm_all = SMOTE(random_state=42)
X_res,y_res = sm_all.fit_resample(X,y)

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model_NB_1_res = DecisionTreeClassifier()
scores1 = cross_val_score(model_NB_1_res, X_res, y_res, scoring='accuracy', cv=cv, n_jobs=-1)
scores2 = cross_val_score(model_NB_1_res, X_res, y_res, scoring='precision', cv=cv, n_jobs=-1)
scores3 = cross_val_score(model_NB_1_res, X_res, y_res, scoring='recall', cv=cv, n_jobs=-1)
scores4 = cross_val_score(model_NB_1_res, X_res, y_res, scoring='f1', cv=cv, n_jobs=-1)
print('Accuracy: %.2f' % (mean(scores1)))
print('Precision: %.2f' % (mean(scores2)))
print('Recall: %.2f' % (mean(scores3)))
print('F1 Score: %.2f' % (mean(scores4)))


# In[470]:


# greatest impact / NB model  (oversampled  data)

weights = pd.Series(importances, index = X.columns.values)
print(weights.sort_values(ascending=False)[:10].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/NB_features_most_BMI30_RES.pdf", bbox_inches='tight')


# In[471]:


# least impact / NB model  (oversampled data)
print(weights.sort_values(ascending=False)[-10:].plot(kind='barh', color='grey'))
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/NB_features_least_BMI30_res.pdf", bbox_inches='tight')


# In[472]:


# roc curve for NB model  (oversampled data)

pred_prob5 = model_NB_res.predict_proba(X_test)
fpr5, tpr5, thresh5 = roc_curve(y_test, pred_prob5[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[473]:


# computing auc score for NB  (oversampled data)
auc_score5 = roc_auc_score(y_test, pred_prob5[:,1])
print(auc_score5)


# In[474]:


# plot roc curves for NB  (oversampled data)

plt.style.use('seaborn')

plt.plot(tpr5, fpr5, linestyle='--',color='red', label='NB')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/ROC_curve_NB_BMI30_res.pdf")
plt.show();


# In[ ]:





# In[475]:


#====================================
# KNN model training (original data)
#====================================

# testing accuracy for k from 1 to 25
k_range = range(1,26)
scores={}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))


# In[476]:


#plotting the relationship between k values and correspondgin testing accuracy (original data)

plt.plot(k_range,scores_list, color='black')
plt.xlabel('K value for KNN')
plt.ylabel('Testing accuracy')


# In[477]:


# the final model should have k=12 (original data)
model_KNN = KNeighborsClassifier(n_neighbors=12)
model_KNN.fit(X_train, y_train)


# In[478]:


#print out the hyperparameters of the trained model (original data)
model_KNN.get_params()


# In[479]:


# KNN model prediction and metrics (original data)

y_pred = model_KNN.predict(X_test)

print('KNN model metrics:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: \t {0:.2f}'.format(metrics.precision_score(y_test, y_pred)))
print('Recall: \t {0:.2f}'.format(metrics.recall_score(y_test, y_pred)))
print('F1-score: \t {0:.2f}'.format(metrics.f1_score(y_test, y_pred)))


# In[480]:


# DT model performance with K-folds validation (original data)

from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model_KNN_1 = KNeighborsClassifier()
scores1 = cross_val_score(model_KNN_1, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores2 = cross_val_score(model_KNN_1, X, y, scoring='precision', cv=cv, n_jobs=-1)
scores3 = cross_val_score(model_KNN_1, X, y, scoring='recall', cv=cv, n_jobs=-1)
scores4 = cross_val_score(model_KNN_1, X, y, scoring='f1', cv=cv, n_jobs=-1)
print('Accuracy: %.2f' % (mean(scores1)))
print('Precision: %.2f' % (mean(scores2)))
print('Recall: %.2f' % (mean(scores3)))
print('F1 Score: %.2f' % (mean(scores4)))


# In[481]:


# feature importance is not defined for the KNN classification algorithm and there is no easy way to calucate it


# In[482]:


# roc curve for KNN model (original data)

pred_prob6 = model_KNN.predict_proba(X_test)
fpr6, tpr6, thresh6 = roc_curve(y_test, pred_prob6[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[483]:


# computing auc score for KNN (original data)
auc_score6 = roc_auc_score(y_test, pred_prob6[:,1])
print(auc_score6)


# In[484]:


# plot roc curve (original data)

plt.style.use('seaborn')


plt.plot(tpr6, fpr6, linestyle='--',color='red', label='KNN')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/ROC_curve_KNN_BMI30.pdf")
plt.show();


# In[ ]:





# In[485]:


#====================================
# KNN model training (oversampled data)
#====================================

# testing accuracy for k from 1 to 25
k_range = range(1,26)
scores={}
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train_res,y_train_res)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(y_test,y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))


# In[486]:


#plotting the relationship between k values and correspondgin testing accuracy (oversampled data)

plt.plot(k_range,scores_list, color='black')
plt.xlabel('K value for KNN')
plt.ylabel('Testing accuracy')


# In[487]:


# the final model should have k=2 (oversampled data)
model_KNN_res = KNeighborsClassifier(n_neighbors=2)
model_KNN_res.fit(X_train_res, y_train_res)


# In[488]:


# KNN model prediction and metrics (oversampled data)

y_pred = model_KNN_res.predict(X_test)

print('KNN model metrics:')
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision: \t {0:.2f}'.format(metrics.precision_score(y_test, y_pred)))
print('Recall: \t {0:.2f}'.format(metrics.recall_score(y_test, y_pred)))
print('F1-score: \t {0:.2f}'.format(metrics.f1_score(y_test, y_pred)))


# In[489]:


# DT model performance with K-folds validation (oversampled data)


#oversampling the whole dataset (not just training data)
sm_all = SMOTE(random_state=42)
X_res,y_res = sm_all.fit_resample(X,y)

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model_KNN_1_res = DecisionTreeClassifier()
scores1 = cross_val_score(model_KNN_1_res, X_res, y_res, scoring='accuracy', cv=cv, n_jobs=-1)
scores2 = cross_val_score(model_KNN_1_res, X_res, y_res, scoring='precision', cv=cv, n_jobs=-1)
scores3 = cross_val_score(model_KNN_1_res, X_res, y_res, scoring='recall', cv=cv, n_jobs=-1)
scores4 = cross_val_score(model_KNN_1_res, X_res, y_res, scoring='f1', cv=cv, n_jobs=-1)
print('Accuracy: %.2f' % (mean(scores1)))
print('Precision: %.2f' % (mean(scores2)))
print('Recall: %.2f' % (mean(scores3)))
print('F1 Score: %.2f' % (mean(scores4)))


# In[490]:


# roc curve for KNN model (oversampled data)

pred_prob6 = model_KNN_res.predict_proba(X_test)
fpr6, tpr6, thresh6 = roc_curve(y_test, pred_prob6[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


# In[491]:


auc_score6 = roc_auc_score(y_test, pred_prob6[:,1])
print(auc_score6)


# In[492]:


# plot roc curve (original data)

plt.style.use('seaborn')


plt.plot(tpr6, fpr6, linestyle='--',color='red', label='KNN')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/ROC_curve_KNN_BMI30_res.pdf")
plt.show();


# In[ ]:





# In[493]:


# a plot of roc curves of all 5 models (oversampled data)
plt.style.use('seaborn')

# plot roc curves
plt.plot(tpr1, fpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(tpr2, fpr2, linestyle='--',color='green', label='Random Forest')
plt.plot(tpr3, fpr3, linestyle='--',color='violet', label='Support Vector Machine')
plt.plot(tpr4, fpr4, linestyle='--',color='black', label='Decision Tree')
plt.plot(tpr5, fpr5, linestyle='--',color='red', label='Naive Bayes')

plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/ROC_curve_all5_BMI30_res.pdf")
plt.show();


# In[494]:


# a plot of roc curves of all 6 models (oversampled data)
plt.style.use('seaborn')

# plot roc curves
plt.plot(tpr1, fpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(tpr2, fpr2, linestyle='--',color='green', label='Random Forest')
plt.plot(tpr3, fpr3, linestyle='--',color='violet', label='Support Vector Machine')
plt.plot(tpr4, fpr4, linestyle='--',color='blue', label='Decision Tree')
plt.plot(tpr5, fpr5, linestyle='--',color='red', label='Naive Bayes')
plt.plot(tpr6, fpr6, linestyle='--',color='brown', label='K-Nearest Neighbor')

plt.plot(p_fpr, p_tpr, linestyle='--', color='black', linewidth=3)
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/ROC_curve_all6_BMI30_res.pdf")
plt.show();


# In[495]:


# a plot of roc curves of all 6 models / black and white
plt.style.use('seaborn')

# plot roc curves
plt.plot(tpr1, fpr1, linestyle=':',color = 'black',label='Logistic Regression')
plt.plot(tpr2, fpr2, linestyle='-',color='black', label='Random Forest')
plt.plot(tpr3, fpr3, linestyle='--',color='black', label='Support Vector Machine')
plt.plot(tpr4, fpr4, linestyle='-.',color='black', label='Decisions Tree')
plt.plot(tpr5, fpr5, linestyle='--',color='black', linewidth=3, label='Naive Bayes')
plt.plot(tpr6, fpr6, linestyle='-.',color='black', label='K-Nearest Neighbor')

plt.plot(p_fpr, p_tpr, linestyle='--', color='black', linewidth=3)
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive Rate')

plt.legend(loc='best')
plt.savefig("C:/Users/Toomas/Desktop/ITMI_Data/Tables and graphs/25_64/ML/ROC_curve_all6_BW_BMI30_res.pdf")
plt.show();


# In[ ]:





# In[ ]:




