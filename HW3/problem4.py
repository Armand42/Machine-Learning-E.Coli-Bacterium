#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:30:01 2019

@author: armandnasserischool
"""

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from scipy import interp
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn.svm import SVC
import pandas as pd
import warnings
np.random.seed(0)
warnings.filterwarnings("ignore")

# Import the dataset 
dataset = pd.read_csv('ecs171.dataset.txt', delim_whitespace=True)


# X features
X = dataset.iloc[:,6:]
# Dropping the last column to avoid type error
X = X.iloc[:, :-1]

def createSVM(X,y):
# y prediction
    print(X.shape)
    newY = dataset[y]
    print(newY.shape)
    newYY = label_binarize(newY, classes=np.unique(newY))
    # Set the different values of alpha to be tested
    alpha_ridge = np.array([1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 0.1,1, 5, 10, 20])

# Applying lasso regression and grid search to determine optimal alpha value
    lasso = linear_model.Lasso()
    grid = GridSearchCV(estimator=lasso, scoring = "neg_mean_squared_error",param_grid = dict(alpha=alpha_ridge), cv = 5)
    grid.fit(X,newYY)
# Inserting optimal lambda value into another lasso regression to reduce the amount of features
    lassoBest = linear_model.Lasso(alpha=grid.best_estimator_.alpha, max_iter=1000)
    lassoBest.fit(X,newYY)
    print("The optimal constrained lambda parameter value is:",grid.best_estimator_.alpha)
# Removin all zero coefficients
    #coeff_used = np.sum(lassoBest.coef_!=0)
   
    #lassoCoef = np.array([])
    #for i in lassoBest.coef_:
     #   if (i != 0):
      #      lassoCoef = np.append(lassoCoef,i)
    #print("NUM OF LASSO COEFFFF IS:",lassoCoef)

# Now have features
    kf = KFold(n_splits=5, random_state=1)
    svm =  OneVsRestClassifier(SVC(C=0.25, kernel='linear', probability = True))
    
    acc_list = []
    
    # This should return the new dataset of features after applying lasso (removes non-zero coefficients too)
    model = SelectFromModel(lassoBest, prefit=True)
    X_new = model.transform(X)
    
    # Something wrong with GenePerturbed for lasso
    print(X_new.shape)
    
    
    #i = 0
    #tprs = []
    #aucs = []
    #mean_fpr = np.linspace(0, 1, 100)
    #for train, test in kf.split(X_new, newYY):
     #    X_train, X_test = X_new[train], X_new[test]
      #   y_train, y_test = newYY[train], newYY[test]
       #  y_train = np.argmax(y_train, axis=1)
        # y_test = np.argmax(y_test, axis=1)
         
         #probas_ = svm.fit(X_train, y_train).decision_function(X_new[test])
         #print(probas_)
         #fpr, tpr, _ = roc_curve(newYY[test].ravel(), probas_.ravel())
         #roc_auc = auc(fpr, tpr)
         #plt.plot(fpr, tpr, alpha=0.2, label='%d fold (AUC: %0.2f)' % (i, roc_auc))
         #i += 1
         #probas_ = svm.fit(X_new[train], newYY[y_train]).predict_proba(X_new[test])
         #fpr, tpr,_= roc_curve(newYY[test], probas_)
         #tprs.append(interp(mean_fpr, fpr, tpr))
         #tprs[-1][0] = 0.0
         #roc_auc = auc(fpr, tpr)
         #aucs.append(roc_auc)
         #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
         #i = i + 1
    #plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    
    #mean_tpr = np.mean(tprs, axis=0)
    #mean_tpr[-1] = 1.0
    #mean_auc = auc(mean_fpr, mean_tpr)
    #std_auc = np.std(aucs)
    #plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
    
    #std_tpr = np.std(tprs, axis=0)
    #tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    #tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
    
    #plt.xlim([-0.05, 1.05])
    #plt.ylim([-0.05, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    #plt.legend(loc="lower right")
    #plt.show()
   
   
    tprs = []
    roc_auc = dict()
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

  
    i = 0
    ## REFERENCE CODE
    for train_index, test_index in kf.split(X_new, newYY):
        print("TRAIN:", train_index, "TEST:", test_index)
        
        X_train, X_test = X_new[train_index], X_new[test_index]
        y_train, y_test = newYY[train_index], newYY[test_index]
        
        #y_train = np.argmax(y_train, axis=1)
        #y_test = np.argmax(y_test, axis=1)
        
        score = svm.fit(X_train, y_train).predict(X_test)#.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test.ravel(), score.ravel())
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, alpha=0.8, label='%d fold (AUC: %0.2f)' % (i, roc_auc))
        i += 1
        print(i)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    #std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$)' % (mean_auc),lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC with 5-Fold Cross Validation')
    plt.legend(loc="lower right")
    plt.show()
        
       
        
    # Compute micro-average ROC curve and ROC area
 
        
        
    
    
    #plt.figure()
    #lw = 2
    #plt.plot(fpr[2], tpr[2], color='darkorange',
    #     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    #plt.legend(loc="lower right")
    #plt.show()
    
        #print(X_train.shape,y_train.shape)
        #svm.fit(X_train, y_train)
        #print("----Start Evaluating----")
        #acc = svm.score(X_test, y_test)
        #acc_list.append(acc)
        #print("Testing Accuracy:", acc)
    #print("Mean testing accuracy:", sum(acc_list) / len(acc_list))

    #y_pred = svm.predict(X_test)
  
   
      
    
   
    

#print(lassoCoef.shape)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#scaler = StandardScaler()

#scaler.fit(X_train)
#sel_ = SelectFromModel(SVC(C=1, kernel='rbf'))
#sel_.fit(scaler.transform(X_train, y_train))
# Need to get actual features from coeff
# apply svm to 127 features
# apply 4 svms
createSVM(X,'Strain')
plt.title('ROC with 5-Fold Cross Validation (Strain)')
createSVM(X,'Medium')
plt.title('ROC with 5-Fold Cross Validation (Medium)')
createSVM(X,'Stress')
plt.title('ROC with 5-Fold Cross Validation (Stress)')
createSVM(X,'GenePerturbed')
plt.title('ROC with 5-Fold Cross Validation (Gene Perturbed')