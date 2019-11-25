#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:20:25 2019

@author: armandnasserischool
"""

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import pandas as pd
import warnings
np.random.seed(0)
warnings.filterwarnings("ignore")

# Import the dataset 
dataset = pd.read_csv('ecs171.dataset.txt', delim_whitespace=True).dropna()

# X features
X = dataset.iloc[:,6:4503]
# Dropping the last column to avoid type error
X = X.iloc[:, :-1]
# y prediction
y = dataset['GrowthRate']

# Set the different values of alpha to be tested
alpha_ridge = np.array([1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20])

# Applying lasso regression and grid search to determine optimal alpha value
lasso = linear_model.Lasso()
grid = GridSearchCV(estimator=lasso, scoring = "neg_mean_squared_error",param_grid = dict(alpha=alpha_ridge), cv = 5)
grid.fit(X,y)
# Inserting optimal lambda value into another lasso regression to reduce the amount of features
lassoBest = linear_model.Lasso(alpha=grid.best_estimator_.alpha, max_iter=10e5)
lassoBest.fit(X,y)
# Removin all zero coefficients
coeff_used = np.sum(lassoBest.coef_!=0)
# Printing out the data
print("The 5-fold cross-validation error is:",-grid.best_score_)
print("The optimal constrained lambda parameter value is:",grid.best_estimator_.alpha)
print("The number of features that have non-zero coefficients for alpha = 0.0001 is:", coeff_used)

#means = grid.cv_results_['mean_test_score']
#stds = grid.cv_results_['std_test_score']
#params = grid.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#    print("%f (%f) with: %r" % (mean, stdev, param))

#it = 0
#for i in alpha_ridge:
 #   lasso = linear_model.Lasso(alpha=i, max_iter=10e5)
  #  lasso.fit(X_train,y_train)
   # print("Iteration:", it)
    #print("")
    #it = it+1

    #train_score=lasso.score(X_train,y_train)
    #test_score=lasso.score(X_test,y_test)
    #coeff_used = np.sum(lasso.coef_!=0)

    #print("training score:", train_score) 
    #print("test score: ", test_score)
    #print("number of features used: ", coeff_used)

#print(cross_val_score(lasso, X, y, cv=5))




        
        



