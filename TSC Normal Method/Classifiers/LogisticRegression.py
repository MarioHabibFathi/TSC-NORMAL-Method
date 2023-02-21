#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:37:43 2023

@author: mariohabibfathi
"""
import numpy as np
from sklearn.metrics import accuracy_score


class LogisticRegression():
    
    def __init__(self,learning_rate=0.1, epochs=150, regularization = None, C=1.0,l_lambda = 0.01):
        self.losses = []
        self.train_accuracies = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.C = C
        self.l_lambda = l_lambda
        self.multiclass = False
        
    def fit(self,x,y):
        
        if len(np.unique(y))>2:
            self.multiclass = True
            self.weights_multi = [ [] for i in range(len(np.unique(y)))]
            self.b_multi = [ [] for i in range(len(np.unique(y)))]
            self.losses = [[] for i in range(len(np.unique(y)))]
            self.train_accuracies = [[] for i in range(len(np.unique(y)))]
            self.classes = np.unique(y)
            for i in range(len(self.classes)):
                self.weights_multi[i] = np.zeros(x.shape[1])
                self.b_multi[i] = 0
                y_by_class = []
                for k in y:
                    if k==i:
                        y_by_class.append(1)
                    else:
                        y_by_class.append(0)

                # print(y_by_class)
                for _ in range (self.epochs):
                    
                    x_dot_pro_w =  np.matmul(self.weights_multi[i], x.transpose()) + self.b_multi[i]
                    pred = self._calculate_sigmoid(x_dot_pro_w)
                    loss = self.loss_calculation(pred, y_by_class,i)
                    self.gradient_descent(x, y_by_class, pred,i)
                    pred_to_class = [1 if p > 0.5 else 0 for p in pred]
                    self.train_accuracies[i].append(accuracy_score(y, pred_to_class))
                    self.losses[i].append(loss)
            
        else:
            
            self.weights = np.zeros(x.shape[1])
            self.b = 0
            
            for _ in range (self.epochs):
                
                x_dot_pro_w =  np.matmul(self.weights, x.transpose()) + self.b
                pred = self._calculate_sigmoid(x_dot_pro_w)
                loss = self.loss_calculation(pred, y)
                self.gradient_descent(x, y, pred)
                pred_to_class = [1 if p > 0.5 else 0 for p in pred]
                self.train_accuracies.append(accuracy_score(y, pred_to_class))
                self.losses.append(loss)
            
            # print(self.weights)
    
    def gradient_descent(self,x,y,y_pred,i=0):
        diff = y_pred - y
        gradient_b = np.mean(diff)
        gradients_w = np.matmul(x.transpose(), diff)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])
        regularization = 0
        if self.multiclass:
            if self.regularization == 'l1':
                regularization = self.l_lambda * np.abs(self.weights_multi[i]).sum()
            elif self.regularization == 'l2':
                regularization = self.l_lambda * np.power(self.weights_multi[i], 2).sum()
            self.weights_multi[i] = self.weights_multi[i] - self.learning_rate * (gradients_w+regularization)
            self.b_multi[i] = self.b_multi[i] - self.learning_rate * gradient_b
        else:
            if self.regularization == 'l1':
                regularization = self.l_lambda * np.abs(self.weights).sum()
            elif self.regularization == 'l2':
                regularization = self.l_lambda * np.power(self.weights, 2).sum()
            self.weights = self.weights - self.learning_rate * (gradients_w+regularization)
            self.b = self.b - self.learning_rate * gradient_b
        
    
    def loss_calculation(self,y_pred,y,i=0):
        regularization = 0
        if self.multiclass:
            if self.regularization == 'l1':
                regularization = self.l_lambda * np.abs(self.weights_multi[i]).sum()
            elif self.regularization == 'l2':
                regularization = self.l_lambda * np.power(self.weights_multi[i], 2).sum()
        else:
            if self.regularization == 'l1':
                regularization = self.l_lambda * np.abs(self.weights).sum()
            elif self.regularization == 'l2':
                regularization = self.l_lambda * np.power(self.weights, 2).sum()
                
            return -np.mean(y*np.log(y_pred)+(1-y) *np.log(y_pred))+regularization
        
    
    def _calculate_sigmoid(self,x):
        
        y = []
        for value in x:
            y.append(self._sigmoid_function(value))
        return np.array(y)
    
    def _sigmoid_function(self,x):
        
        if x >= 0:
            return 1/(1+np.exp(-x))
        else:
            return np.exp(x)/(1+np.exp(x))
    
    
   
    def predict(self,x,threshold = 0.5):
        pred = []
        if self.multiclass:
            pred_by_class=[[] for i in range(len(self.classes))]
            # print(np.array(pred_by_class).shape)
            for i in range(len(self.classes)):
                x_w = np.matmul(x, self.weights_multi[i].transpose()) + self.b_multi[i]
                prob= self._calculate_sigmoid(x_w)
                pred_by_class[i].append(prob)
            # print(np.array(pred_by_class).shape)
            
            # return np.array(pred_by_class)
            for j in range(len(x)):
                max_prob = pred_by_class[0][0][j]
                predicted_class = 0
                # print(max_prob)
                # print(np.array(max_prob).shape)
                for i in range(len(self.classes)):
                    # print("before")
                    val = pred_by_class[i][0][j]
                    # print(val)
                    if val > max_prob:
                        predicted_class = i
                pred.append(predicted_class)
            # print(pred_by_class)
            
            # return pred_by_class
        else:           
            x_w = np.matmul(x, self.weights.transpose()) + self.b
            probabilities = self._calculate_sigmoid(x_w)
            for p in probabilities:
                if p > threshold:
                    pred.append(1)
                else:
                    pred.append(0)
        return np.array(pred)
    
import tensorflow as tf

# class LogisticRegression:
#     EPS = 1e-5
#     def __ols_solve(self, x, y):
#         rows, cols = x.shape
#         if rows >= cols == tf.linalg.matrix_rank(x):
#             y = tf.math.maximum(self.EPS, tf.math.minimum(tf.cast(y, tf.float32), 1-self.EPS))
#             ols_y = -tf.math.log(tf.math.divide(1, y) - 1)
#             self.weights = tf.linalg.matmul(
#                 tf.linalg.matmul(
#                     tf.linalg.inv(
#                         tf.linalg.matmul(x, x, transpose_a=True)
#                     ),
#                     x, transpose_b=True),
#                 ols_y)
#         else:
#             print('Error! X has not full column rank.')
    
#     def __sgd(self, x, y, loss_fn, learning_rate, iterations, batch_size):
#         rows, cols = x.shape
#         self.weights = tf.Variable(tf.random.normal(stddev=1.0/cols, shape=(cols, 1)))
#         dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
        
#         for i in range(iterations):
#             dataset.shuffle(buffer_size=1024)
#             for step, (xb, yb) in enumerate(dataset):
#                 with tf.GradientTape() as tape:
#                     loss = loss_fn(xb, yb)
#                 grads = tape.gradient(loss, self.weights)
#                 self.weights.assign_sub(learning_rate*grads)
    
#     def __sse_loss(self, xb, yb):
#         yb = tf.math.maximum(self.EPS, tf.math.minimum(tf.cast(yb, tf.float32), 1-self.EPS))
#         ols_yb = -tf.math.log(tf.math.divide(1, yb) - 1)
        
#         diff = tf.linalg.matmul(xb, self.weights) - ols_yb
#         loss = tf.linalg.matmul(diff, diff, transpose_a=True)
        
#         return loss
    
#     def __mle_loss(self, xb, yb):
#         xw = tf.linalg.matmul(xb, self.weights)
#         term1 = tf.linalg.matmul(tf.cast(1-yb, tf.float32), xw, transpose_a=True)
#         term2 = tf.linalg.matmul(
#             tf.ones_like(yb, tf.float32),
#             tf.math.log(1+tf.math.exp(-xw)),
#             transpose_a=True)
#         return term1+term2
    
#     def fit(self, x, y, method, learning_rate=0.001, iterations=500, batch_size=32):
#         x = tf.concat([x, tf.ones_like(y, dtype=tf.float32)], axis=1)
#         if method == "ols_solve":
#             self.__ols_solve(x, y)
#         elif method == "ols_sgd":
#             self.__sgd(x, y, self.__sse_loss, learning_rate, iterations, batch_size)
#         elif method == "mle_sgd":
#             self.__sgd(x, y, self.__mle_loss, learning_rate, iterations, batch_size)
#         else:
#             print(f'Unknown method: \'{method}\'')
        
#         return self
    
#     def predict(self, x):
#         if not hasattr(self, 'weights'):
#             print('Cannot predict. You should call the .fit() method first.')
#             return
        
#         x = tf.concat([x, tf.ones((x.shape[0], 1), dtype=tf.float32)], axis=1)
        
#         if x.shape[1] != self.weights.shape[0]:
#             print(f'Shapes do not match. {x.shape[1]} != {self.weights.shape[0]}')
#             return
        
#         xw = tf.linalg.matmul(x, self.weights)
#         return tf.math.divide(1, 1+tf.math.exp(-xw))
    
#     def accuracy(self, x, y):
#         y_hat = self.predict(x)
        
#         if y.shape != y_hat.shape:
#             print('Error! Predictions don\'t have the same shape as given y')
#             return
        
#         zeros, ones = tf.zeros_like(y), tf.ones_like(y)
#         y = tf.where(y >= 0.5, ones, zeros)
#         y_hat = tf.where(y_hat >= 0.5, ones, zeros)
        
#         return tf.math.reduce_mean(tf.cast(y == y_hat, tf.float32))
    
    
    
    
    
    
    
    
