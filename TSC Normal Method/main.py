#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:37:03 2023

@author: mariohabibfathi
"""


# import Constants
from Constants import *

import Datasets as ds
import Distance_Metrics as DM
import visualization as VS
import Classifiers.classifiers as CL
import Classifiers.LogisticRegression as LR

from sklearn.metrics import accuracy_score
import pandas as pd

from sklearn.preprocessing import LabelEncoder

# from sklearn import svm
from sklearn.linear_model import LogisticRegression
import numpy as np



d = ds.Dataset(UCR_PATH)

# d.Show_Dataset()

# datasets = ['SmoothSubspace','SyntheticControl','BeetleFly','BirdChicken',
#             'Coffee','UMD','Meat','ItalyPowerDemand','Chinatown','ECG200']

datasets = ['SmoothSubspace']

# xtrain,ytrain,xtest,ytest = d.load_dataset(datasets[0])

# xtrain = d.znormalisation(xtrain)
# xtest = d.znormalisation(xtest)

# LE = LabelEncoder()
    
# ytrain = LE.fit_transform(ytrain)
# ytest = LE.fit_transform(ytest)

# v = VS.Dataset_Visualization()
# v.Plot_all_by_Class(xtrain, ytrain)
acc = []
err_rate = []

for da in datasets:
    
    xtrain,ytrain,xtest,ytest = d.load_dataset(da)
    
    xtrain = d.znormalisation(xtrain)
    xtest = d.znormalisation(xtest)
    
    LE = LabelEncoder()
    
    ytrain = LE.fit_transform(ytrain)
    ytest = LE.fit_transform(ytest)
    
    
    # print(np.unique(ytrain))
    
    # model = LogisticRegression(solver='sag', max_iter=150)
    # model.fit(xtrain, ytrain)
    # pred = model.predict(xtest)
    
    lr = LR.LogisticRegression(regularization='l1')
    lr.fit(xtrain, ytrain)
    pred = lr.predict(xtest)
    print(accuracy_score(ytest, pred))
    
    # xxxxxxxxxxxxxxxx
    # sk = svm.SVC()

    # sk.fit(xtrain, ytrain)
    # pred = sk.predict(xtest)
    # SVM = CL.SVM()
    
    # pred = NN.k_nearest_neighbor(xtrain, ytrain, xtest, metric_distance='DTW')
    
    # SVM.fit(xtrain, ytrain)

    # pred = SVM.predict(xtest)

    # acc.append(accuracy_score(ytest, pred))
    # err_rate.append(1-accuracy_score(ytest, pred))

    # print("dsds")




# print(1.0 - accuracy_score(ytest, pred))
# df = pd.DataFrame(list(zip(datasets,acc,err_rate)),columns = ['Dataset','accuracy','Error rate'])
# df.to_csv('outputs/LR/Test me/result_ovr_no_reg.csv')

# VS.Dataset_Visualization().Plot_all(xtrain,ytrain,save=True)
# VS.Dataset_Visualization().Plot_all_by_Class(xtrain,ytrain,save=True)

# VS.Dataset_Visualization().Plot_by_Class(xtrain,ytrain,save=True)

# VS.Dataset_Visualization().Plot_individual(xtrain,ytrain,save=True)