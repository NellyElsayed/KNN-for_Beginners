# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 16:14:04 2021

@author: elsayeny
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

#lets's import the iris dataset
iris = datasets.load_iris()

X = iris.data # X contains the data from iris dataset only (features)
y= iris.target

#extract columo from data
feature1 = X[:,0]
print(feature1.shape)

