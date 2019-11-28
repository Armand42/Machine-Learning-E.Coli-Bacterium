#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:39:26 2019

@author: armandnasserischool
"""

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from matplotlib import pyplot
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


X_mean = np.mean(X, axis=0)
X_mean = X_mean.values.reshape(1,-1)
y_mean = np.array([np.mean(y,axis = 0)])



# configure bootstrap
n_iterations = 300
n_size = int(len(X) * 0.50)
# run bootstrap
stats = list()


for i in range(n_iterations):
# prepare train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
    
    model = linear_model.Lasso(alpha=grid.best_estimator_.alpha)
    model.fit(X_train, y_train)
	# evaluate model
    predictions = model.predict(X_mean)
    
 
    score = mean_squared_error(y_mean,predictions)
    print(score)
    stats.append(score)
# plot scores
pyplot.hist(stats)
pyplot.title("Confidence Interval (Bootstrapping Method)")
pyplot.xlabel("Mean Squared Error")
pyplot.ylabel("Number of samples")
pyplot.show()
# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print('%.1f confidence interval %.5f%% and %.5f%%' % (alpha*100, lower*100, upper*100))



