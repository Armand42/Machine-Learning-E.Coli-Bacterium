#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:29:44 2019

@author: armandnasserischool
"""

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error 
from matplotlib import pyplot
import warnings
np.random.seed(0)
warnings.filterwarnings("ignore")

# Import the dataset 
dataset = pd.read_csv('ecs171.dataset.txt', delim_whitespace=True)

# X features
X = dataset.iloc[:,6:4503]
# Dropping the last column to avoid type error
X = X.iloc[:, :-1]
# y prediction
y = dataset['GrowthRate']

# Set the different values of alpha to be tested
alpha_lasso = np.array([1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20])

# Applying lasso regression and grid search to determine optimal alpha value
lasso = linear_model.Lasso()
grid = GridSearchCV(estimator=lasso, scoring = "neg_mean_squared_error",param_grid = dict(alpha=alpha_lasso), cv = 5)
grid.fit(X,y)
# Inserting optimal lambda value into another lasso regression to reduce the amount of features
lassoBest = linear_model.Lasso(alpha=grid.best_estimator_.alpha, max_iter=10e5)
lassoBest.fit(X,y)
# Removin all zero coefficients
coeff_used = np.sum(lassoBest.coef_!=0)

# Taking the mean of dataset and predictor 
X_mean = np.mean(X, axis=0)
X_mean = X_mean.values.reshape(1,-1)
y_mean = np.array([np.mean(y,axis = 0)])

# Configure bootstrap
num_iterations = 300
stats = list()

for i in range(num_iterations):
     # prepare train and test sets for dataset sampling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # fitting the model to lasso regression
    model = linear_model.Lasso(alpha=grid.best_estimator_.alpha)
    model.fit(X_train, y_train)
	# evaluate model
    predictions = model.predict(X_mean)
    # calculate score
    score = mean_squared_error(y_mean,predictions)
    stats.append(score)
# plot scores
pyplot.hist(stats)
pyplot.title("Confidence Interval (Bootstrapping Method)")
pyplot.xlabel("Mean Squared Error")
pyplot.ylabel("Number of samples")
pyplot.show()
# Calculating confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print('%.1f confidence interval %.5f%% and %.5f%%' % (alpha*100, lower*100, upper*100))