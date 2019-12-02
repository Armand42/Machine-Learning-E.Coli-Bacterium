#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:00:08 2019

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
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import seaborn as sns
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
X = StandardScaler().fit_transform(X)

yStrain = dataset['Strain']
#yMedium = dataset['Medium']
#print(set(yStrain))
#print(set(yMedium))

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

newX = pd.concat([principalDf, dataset[['Strain']]], axis = 1)
plt.title("PCA Plot")
sns.scatterplot(x = newX['principal component 1'], y = newX['principal component 2'],hue=newX['Strain'])

X_embedded = TSNE(n_components=2).fit_transform(X)
tsneDf = pd.DataFrame(data = X_embedded, columns = ['tsne 1', 'tsne 2'])
newX2 = pd.concat([tsneDf, dataset[['Strain']]], axis = 1)

#plt.title("TSNE Plot")
#sns.scatterplot(x = newX2['tsne 1'], y = newX2['tsne 2'],hue=newX2['Strain'])

# AM I DOING THIS RIGHT??
