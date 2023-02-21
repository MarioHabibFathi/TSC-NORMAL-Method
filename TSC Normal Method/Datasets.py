#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:43:09 2023

@author: mariohabibfathi
"""

# import Constants
import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import cm


class Dataset:

    def __init__(self,path):
        self.path = path
        
        
    def load_dataset(self,Dataset_name):
        dataset_path = self.path+Dataset_name+"/"


        if(not os.path.exists(dataset_path)):
            print ("This {} dataset name does not exist please check the spelling or that this dataset existe".format(Dataset_name))


        train_path = dataset_path + Dataset_name + "_TRAIN.tsv"
        test_path = dataset_path + Dataset_name + "_TEST.tsv"

        if (not os.path.exists(train_path)):
            print ("This {} dataset does not have a training file".format(Dataset_name))
            Xtrain = Ytrain = None
        else:
            train_df = pd.read_csv(train_path, sep='\t',header=None)
            Xtrain = np.array(train_df.drop(train_df.columns[0],axis = 1))
            Ytrain = train_df.iloc[:,0]

        if (not os.path.exists(test_path)):
            print ("This {} dataset does not have a test file".format(Dataset_name))
            Xtest = Ytest = None

        else:
            test_df = pd.read_csv(test_path, sep='\t',header=None)
            Xtest = np.array(test_df.drop(test_df.columns[0],axis = 1))
            Ytest = test_df.iloc[:,0]
            
            
        return Xtrain,Ytrain,Xtest,Ytest



    def Show_Dataset (self):
        for Dataset in os.scandir(self.path):
            if Dataset.is_dir():
                print(Dataset.path.split("/")[-1])
                
    def znormalisation(self,x):
    
        stds = np.std(x,axis=1,keepdims=True)
        if len(stds[stds == 0.0]) > 0:
            stds[stds == 0.0] = 1.0
            return (x - x.mean(axis=1, keepdims=True)) / stds
        return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True))




