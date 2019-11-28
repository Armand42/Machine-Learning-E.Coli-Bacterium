#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:30:01 2019

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

lassoCoef = np.array([])
for i in lassoBest.coef_:
    if (i != 0):
        lassoCoef = np.append(lassoCoef,i)

# Need to get actual features
