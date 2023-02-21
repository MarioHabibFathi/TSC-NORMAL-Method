#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:54:18 2023

@author: mariohabibfathi
"""
import numpy as np
import math

class Distance_metrics:
    

    def __init__(self,metric_distance='DTW', window=3):
        
        self.metric_distance = metric_distance
        self.window = window
        
        self.metrics = {
            'DTW' : self.DTW,
            'ED' : self.Euclidean_distance,
            'WDTW' : self.DTW_Windowed
            }

    def apply_metric(self, s1, s2):
        
        return self.metrics[self.metric_distance](s1=s1, s2=s2, window=self.window)

    def DTW (self,s1,s2, window=None):
        
        n = len (s1)
        m = len(s2)    
        dtw_matrix = np.full((n+1, m+1), np.inf)
        dtw_matrix[0,0] = 0
        for i in range (1,n+1):
            for j in range (1,m+1):
                dtw_matrix[i,j] = (s1[i-1] - s2[j-1])**2 + min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
        return math.sqrt(dtw_matrix[n,m])
        
    def DTW_Windowed (self,s1,s2,window=3):
        n,m = len (s1),len(s2)    
        w = np.max([window,abs(n-m)])
        dtw_matrix = np.full((n+1, m+1), np.inf)
        dtw_matrix[0,0] = 0
        for i in range(1, n+1):
            for j in range(np.max([1, i-w]), np.min([m, i+w])+1):
                dtw_matrix[i, j] = 0
        for i in range (1,n+1):
            for j in range (np.max([1, i-w]),np.min([m, i+w])+1):
                dtw_matrix[i,j] = abs(s1[i-1] - s2[j-1]) + min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
        return dtw_matrix[n,m]
    
    def Euclidean_distance(self,s1,s2, window=None):
        return np.sqrt(np.sum((s1-s2)**2))
    