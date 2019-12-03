#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:00:08 2019

@author: armandnasserischool
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
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

def plotPCA(X):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
    
    newX = pd.concat([principalDf, dataset[['Strain']]], axis = 1)
    plt.title("PCA Plot")
    
    sns.scatterplot(x = newX['principal component 1'], y = newX['principal component 2'])
   
def plotTSNE(X):
    X_embedded = TSNE(n_components=2).fit_transform(X)
    tsneDf = pd.DataFrame(data = X_embedded, columns = ['tsne 1', 'tsne 2'])
    
    newX2 = pd.concat([tsneDf, dataset[['Strain']]], axis = 1)
    plt.title("TSNE Plot")
   
    sns.scatterplot(x = newX2['tsne 1'], y = newX2['tsne 2'])

# Plotting PCA and TSNE Plots
plotPCA(X)
plt.figure()
plotTSNE(X)
