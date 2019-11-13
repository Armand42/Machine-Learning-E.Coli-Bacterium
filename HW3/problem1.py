#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:20:25 2019

@author: armandnasserischool
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Import the dataset 
dataset = pd.read_csv('ecs171.dataset.txt', delim_whitespace=True)
# X features 