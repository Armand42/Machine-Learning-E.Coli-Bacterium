#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:00:16 2019

@author: armandnasserischool
"""
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from scipy import interp
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
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
# Creates an svm with roc and pr curves

# Creates an svm with roc and pr curves
def buildSVM(X,y,rocCurveName,method):
    # y prediction
    newY = dataset[y]
    # Need to binary encode categorical variable
    newYY = label_binarize(newY, classes=np.unique(newY))
    # Set the different values of alpha to be tested
    alpha_lasso = np.array([1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 0.1,1, 5, 10, 20])
    # Applying lasso regression and grid search to determine optimal alpha value
    lasso = linear_model.Lasso()
    grid = GridSearchCV(estimator=lasso, scoring = "neg_mean_squared_error",param_grid = dict(alpha=alpha_lasso), cv = 5)
    grid.fit(X,newYY)
    # Inserting optimal lambda value into another lasso regression to reduce the amount of features
    lassoBest = linear_model.Lasso(alpha=grid.best_estimator_.alpha, max_iter=1000)
    lassoBest.fit(X,newYY)

    # Creating a 10-fold cross validation object
    kf = KFold(n_splits=10, random_state=1)
    # Need One vs Rest to handle multiclass shaping issue
    svm =  OneVsRestClassifier(SVC(C=0.25, kernel='linear', probability = True))
    # Returns the new dataset of features after applying PCA
    if (method == 'pca'):
        model = PCA(n_components=2)
        X_new = model.fit_transform(X)
    elif(method == 'tsne'):
        X_new = TSNE(n_components=2).fit_transform(X)
        
  
    # vectors to hold results
    tprs = []
    roc_auc = dict()
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    # Example taken and modified from sklearn
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    i = 0
    # Applying 10-fold cross validation and plotting ROC/PR Curves
    for train_index, test_index in kf.split(X_new, newYY):
        # Splitting up train and test data samples
        X_train, X_test = X_new[train_index], X_new[test_index]
        y_train, y_test = newYY[train_index], newYY[test_index]
        # Fitting svm to training data to predict test
        score = svm.fit(X_train, y_train).predict(X_test)
        # Fitting the ROC curve
        fpr, tpr, _ = roc_curve(y_test.ravel(), score.ravel())
        
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, alpha=0.8, label='%d fold (AUC: %0.2f)' % (i, roc_auc))
        i += 1
    # Plotting the AUC and AUPRC values
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    
    plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$)' % (mean_auc),lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
    # ROC plot
    plt.title(rocCurveName)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    



def buildSVM2(X,y,prCurveName, method):
    # y prediction
    newY = dataset[y]
    # Need to binary encode categorical variable
    newYY = label_binarize(newY, classes=np.unique(newY))
    # Set the different values of alpha to be tested
    alpha_lasso = np.array([1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 0.1,1, 5, 10, 20])
    # Applying lasso regression and grid search to determine optimal alpha value
    lasso = linear_model.Lasso()
    grid = GridSearchCV(estimator=lasso, scoring = "neg_mean_squared_error",param_grid = dict(alpha=alpha_lasso), cv = 5)
    grid.fit(X,newYY)
    # Inserting optimal lambda value into another lasso regression to reduce the amount of features
    lassoBest = linear_model.Lasso(alpha=grid.best_estimator_.alpha, max_iter=1000)
    lassoBest.fit(X,newYY)

    # Creating a 10-fold cross validation object
    kf = KFold(n_splits=10, random_state=1)
    # Need One vs Rest to handle multiclass shaping issue
    svm =  OneVsRestClassifier(SVC(C=0.25, kernel='linear', probability = True))
    # Returns the new dataset of features after applying PCA
    if (method == 'pca'):
        model = PCA(n_components=2)
        X_new = model.fit_transform(X)
    elif(method == 'tsne'):
        X_new = TSNE(n_components=2).fit_transform(X)
        
  
    model = SelectFromModel(lassoBest, prefit=True)
    X_new = model.transform(X)
  
    # vectors to hold results
    tprs = []
    roc_auc = dict()
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    # Example taken and modified from sklearn
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    i = 0
    # Applying 5-fold cross validation and plotting ROC/PR Curves
    for train_index, test_index in kf.split(X_new, newYY):
        # Splitting up train and test data samples
        X_train, X_test = X_new[train_index], X_new[test_index]
        y_train, y_test = newYY[train_index], newYY[test_index]
        # Fitting svm to training data to predict test
        score = svm.fit(X_train, y_train).predict(X_test)
        # Fitting the PR curve
        precision, recall, _ = precision_recall_curve(y_test.ravel(), score.ravel())
        
        tprs.append(interp(mean_fpr, precision, recall))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)
        roc_auc = average_precision_score(y_test.ravel(), score.ravel())
        plt.plot(precision, recall, alpha=0.8, label='%d fold (AUC: %0.2f)' % (i, roc_auc))
        i += 1
    # Plotting the AUC and AUPRC values
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    
    plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean Combined ROC (AUPRC = %0.2f $\pm$)' % (mean_auc),lw=2, alpha=.8)

  
    # PR plot
    plt.title(prCurveName)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc="lower right")
    plt.show()
    
    
    
    
# SVM for PCA
buildSVM(X,'Strain','ROC with 10-Fold Cross Validation (Strain) & PCA', 'pca') 
buildSVM(X,'Medium','ROC with 10-Fold Cross Validation (Medium) & PCA', 'pca') 
buildSVM(X,'Stress','ROC with 10-Fold Cross Validation (Stress) & PCA', 'pca') 
buildSVM(X,'GenePerturbed','ROC with 10-Fold Cross Validation (Gene Perturbed) & PCA', 'pca')     
   
# SVM for TSNE
buildSVM(X,'Strain','ROC with 10-Fold Cross Validation (Strain) & TSNE', 'tsne') 
buildSVM(X,'Medium','ROC with 10-Fold Cross Validation (Medium) & TSNE', 'tsne') 
buildSVM(X,'Stress','ROC with 10-Fold Cross Validation (Stress) & TSNE', 'tsne') 
buildSVM(X,'GenePerturbed','ROC with 10-Fold Cross Validation (Gene Perturbed) & TSNE', 'tsne')   
    
# SVM for PCA  
buildSVM2(X,'Strain','Precision Recall (Strain) & PCA', 'pca') 
buildSVM2(X,'Medium','Precision Recall (Medium) & PCA', 'pca') 
buildSVM2(X,'Stress','Precision Recall (Stress) & PCA', 'pca') 
buildSVM2(X,'GenePerturbed','Precision Recall (Gene Perturbed) & PCA', 'pca') 
 
# SVM for TSNE
buildSVM2(X,'Strain','Precision Recall (Strain) & TSNE', 'tsne')   
buildSVM2(X,'Medium','Precision Recall (Medium) & TSNE', 'tsne') 
buildSVM2(X,'Stress','Precision Recall (Stress) & TSNE', 'tsne') 
buildSVM2(X,'GenePerturbed','Precision Recall (Gene Perturbed) & TSNE', 'tsne')    
    
    



