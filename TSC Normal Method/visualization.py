#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 17:52:52 2023

@author: mariohabibfathi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

class Dataset_Visualization:
    
    
    
    def Plot_all(self,X,Y,title = 'All time series collected together',xtitle ='',ytitle =''
                 ,save = False,saveformat = '.pdf',path = 'Outputs/images/'):
        
        color = cm.rainbow(np.linspace(0, 1, len(Y)))        
        for i, c in zip(range(len(Y)), color):
            plt.plot(X[i], color=c)
        plt.title(title)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        if save:
            plt.savefig(path+title+saveformat)
        plt.show()   

    def Plot_all_by_Class(self,X,Y,title = 'All time series sorting class by color',xtitle ='',ytitle =''
                 ,save = False,saveformat = '.pdf',path = 'Outputs/images/'):
        
        if  isinstance(Y,np.ndarray):
            Y_unique_class_values = np.unique(Y)
            Y_values = np.unique(Y)
        else:
            Y_unique_class_values = Y.unique()
            Y_values = Y.unique()
        for i in range(len(Y_unique_class_values)):
            Y_unique_class_values[i] = i
    
        
        train_dict = {}
        for key, i in zip(Y_values,Y_unique_class_values):
            train_dict[key] = i    
        for k, v in train_dict.items(): 
            Y[Y==k] = v
            
        color = cm.rainbow(np.linspace(0, 1, len(Y_unique_class_values)))
        for i in range(len(Y)):
            plt.plot(X[i], color=color[Y[i]],label=Y[i])
        plt.title(title)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        if save:
            plt.savefig(path+title+saveformat)
        plt.show()   

    def Plot_by_Class(self,X,Y,title = 'All time series for class',xtitle ='',ytitle =''
                 ,save = False,saveformat = '.pdf',path = 'Outputs/images/'):
        
        if  isinstance(Y,np.ndarray):
            Y_unique_class_values = np.unique(Y)
        else:
            Y_unique_class_values = Y.unique()
        m = 0
        for k in Y_unique_class_values:
            
            for i in range(len(Y)):
                if Y[i] == k: 
                    m+=1
            color = cm.rainbow(np.linspace(0, 1, m))
            m=0
            for i in range(len(Y)):
                if Y[i] == k:    
                    plt.plot(X[i], color=color[m])
                    m +=1 
            plt.title(title +' {}'.format(k))
            plt.xlabel(xtitle)
            plt.ylabel(ytitle)
            if save:
                plt.savefig(path+title+' {}'.format(k)+saveformat)
            plt.show()     
        
    def Plot_individual(self,X,Y,title = 'Time series in dataset number ',xtitle ='',ytitle =''
                     ,save = False,saveformat = '.pdf',path = 'Outputs/images/'):

        for i in range(len(Y)):
            plt.plot(X[i], color="blue")
            plt.title(title + str(i+1))
            plt.xlabel(xtitle)
            plt.ylabel(ytitle)
            if save:
                plt.savefig(path+title+' {}'.format(i+1)+saveformat)
            plt.show()
