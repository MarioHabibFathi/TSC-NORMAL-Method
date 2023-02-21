#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 19:05:21 2023

@author: mariohabibfathi
"""

import numpy as np
import Distance_Metrics as DM
import copy
from sklearn.metrics import accuracy_score



class classification_methods:
    
    
    def k_nearest_neighbor(self,train,labels,test,k=1,metric_distance = 'DTW'):
        Pred = []
        
        Distanceclass = DM.Distance_metrics(metric_distance)
        
        
        distance_label = []
        min_distance_list = []
        min_distance = np.inf
        label = labels[0]
        
        
        for series in test:
            for i in range (len(train)):
                Distance = Distanceclass.apply_metric(series, train[i])
                if k == 1:
                    if (Distance < min_distance):
                        min_distance = Distance
                        label = labels[i]
                else:
                    if (len(distance_label)<=k):
                        distance_label.append(labels[i])
                        min_distance_list.append(Distance)
                    else:
                        max_distance = max(min_distance_list)
                        if (Distance < max_distance):
                            distance_label[min_distance_list.index(max(min_distance_list))]=labels[i]
                            min_distance_list[min_distance_list.index(max(min_distance_list))]=Distance
                    label = max(distance_label,key=distance_label.count)

                            
            Pred.append(label)
            distance_label = []
            min_distance_list = []
            min_distance = np.inf
        
        return Pred
        
    
class SVM:

    def __init__(self, C = 1.0):
        # C = error term
        self.C = C
        self.w = 0
        self.b = 0
        
    
    def hingeloss(self, w, b, x, y):
        # Regularizer term
        reg = 0.5 * (w * w)
        
        for i in range(x.shape[0]):
            # Optimization term
            opt_term = y[i] * ((np.dot(w, x[i])) + b)
            
            # calculating loss
            loss = reg + self.C * max(0, 1-opt_term)
        return loss[0][0]

        
    def fit(self, X, Y, batch_size=100, learning_rate=0.001, epochs=1000):
        # The number of features in X
        number_of_features = X.shape[1]

        # The number of Samples in X
        number_of_samples = X.shape[0]

        c = self.C

        # Creating ids from 0 to number_of_samples - 1
        ids = np.arange(number_of_samples)

        # Shuffling the samples randomly
        np.random.shuffle(ids)

        # creating an array of zeros
        w = np.zeros((1, number_of_features))
        b = 0
        losses = []

        # Gradient Descent logic
        for i in range(epochs):
            # Calculating the Hinge Loss
            l = self.hingeloss(w, b, X, Y)

            # Appending all losses 
            losses.append(l)
            
            # Starting from 0 to the number of samples with batch_size as interval
            for batch_initial in range(0, number_of_samples, batch_size):
                gradw = 0
                gradb = 0

                for j in range(batch_initial, batch_initial + batch_size):
                    if j < number_of_samples:
                        x = ids[j]
                        ti = Y[x] * (np.dot(w, X[x].T) + b)

                        if ti > 1:
                            gradw += 0
                            gradb += 0
                        else:
                            # Calculating the gradients

                            #w.r.t w 
                            gradw += c * Y[x] * X[x]
                            # w.r.t b
                            gradb += c * Y[x]

                # Updating weights and bias
                w = w - learning_rate * w + learning_rate * gradw
                b = b + learning_rate * gradb
        
        self.w = w
        self.b = b

        return self.w, self.b, losses  

        
    def predict(self, X):
        
        prediction = np.dot(X, self.w[0]) + self.b # w.x + b
        return np.sign(prediction)
        
class LogisticRegression():
    def __init__(self,learning_rate=0.1):
        self.losses = []
        self.train_accuracies = []
        self.learning_rate = learning_rate

    def fit(self, x, y, epochs):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for i in range(epochs):
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)

    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    def compute_gradients(self, x, y_true, y_pred):
        # derivative of binary cross entropy
        difference =  y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b

    def update_model_parameters(self, error_w, error_b):
        self.weights = self.weights - self.learning_rate * error_w
        self.bias = self.bias - self.learning_rate * error_b

    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def _transform_x(self, x):
        x = copy.deepcopy(x)
        return x

    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.reshape(y.shape[0], 1)


# class lrd:
#         def __init__(self, lr=0.1, num_iter=100000, fit_intercept = True, penalty = None, C=1.0):
#             self.lr = lr
#             self.num_iter = num_iter
#             self.fit_intercept = fit_intercept
#             self.penalty = penalty
#             self.C = C
        
#         def sigmoid(self, z):
#             return 1 / (1 + np.exp(-z))
        
#         def add_intercept(self, X):
#             intercept = np.ones((X.shape[0], 1))
#             return np.concatenate((intercept, X), axis=1)
        
#         def loss(self, h, y):
#             return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        
#         def fit(self, X, y):
#             if self.fit_intercept:
#                 X = self.add_intercept(X)
            
#             self.theta = np.zeros(X.shape[1])
            
#             for i in range(self.num_iter):
#                 z = np.dot(X, self.theta)
#                 h = self.sigmoid(z)
#                 gradient = np.dot(X.T, (h - y)) / y.size
                
#                 if self.penalty == 'l1':
#                     gradient[1:] += -1.0 * (self.theta[1:] / self.C)
#                 elif self.penalty == 'l2':
#                     gradient[1:] += -2.0 * (self.theta[1:] / self.C) * self.theta[1:]
                
#                 self.theta -= self.lr * gradient
                
#                 if(i % 10000 == 0):
#                     z = np.dot(X, self.theta)
#                     h = self.sigmoid(z)
#                     # print(f'loss: {self.loss(h, y)} \t')
        
#         def predict_prob(self, X):
#             if self.fit_intercept:
#                 X = self.add_intercept(X)
            
#             return self.sigmoid(np.dot(X, self.theta))
        
#         def predict(self, X, threshold=0.5):
#             return self.predict_prob(X) >= threshold


# class LogisticRegression(object):
#     """
#     Logistic Regression Classifier
#     Parameters
#     ----------
#     learning_rate : int or float, default=0.1
#         The tuning parameter for the optimization algorithm (here, Gradient Descent) 
#         that determines the step size at each iteration while moving toward a minimum 
#         of the cost function.
#     max_iter : int, default=100
#         Maximum number of iterations taken for the optimization algorithm to converge
    
#     penalty : None or 'l2', default='l2'.
#         Option to perform L2 regularization.
#     C : float, default=0.1
#         Inverse of regularization strength; must be a positive float. 
#         Smaller values specify stronger regularization. 
#     tolerance : float, optional, default=1e-4
#         Value indicating the weight change between epochs in which
#         gradient descent should terminated. 
#     """

#     def __init__(self, learning_rate=0.1, max_iter=100, regularization='l2', C = 0.1, tolerance = 1e-4):
#         self.learning_rate  = learning_rate
#         self.max_iter       = max_iter
#         self.regularization = regularization
#         self.C              = C
#         self.tolerance      = tolerance
    
#     def fit(self, X, y):
#         """
#         Fit the model according to the given training data.
#         Parameters
#         ----------
#         X : {array-like, sparse matrix} of shape (n_samples, n_features)
#             Training vector, where n_samples is the number of samples and
#             n_features is the number of features.
#         y : array-like of shape (n_samples,)
#             Target vector relative to X.
#         Returns
#         -------
#         self : object
#         """
#         self.theta = np.zeros(X.shape[1] + 1)
#         X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

#         for _ in range(self.max_iter):
        
#             errors = (self.__sigmoid(X @ self.theta)) - y
#             N = X.shape[1]

#             if self.regularization is not None:
#                 delta_grad = self.learning_rate * ((self.C * (X.T @ errors)) + np.sum(self.theta))
#             else:
#                 delta_grad = self.learning_rate * (X.T @ errors)

#             if np.all(abs(delta_grad) >= self.tolerance):
#                 self.theta -= delta_grad / N
#             else:
#                 break
                
#         return self

#     def predict_proba(self, X):
#         """
#         Probability estimates for samples in X.
#         Parameters
#         ----------
#         X : array-like of shape (n_samples, n_features)
#             Vector to be scored, where `n_samples` is the number of samples and
#             `n_features` is the number of features.
#         Returns
#         -------
#         probs : array-like of shape (n_samples,)
#             Returns the probability of each sample.
#         """
#         return self.__sigmoid((X @ self.theta[1:]) + self.theta[0])
    
#     def predict(self, X):
#         """
#         Predict class labels for samples in X.
#         Parameters
#         ----------
#         X : array_like or sparse matrix, shape (n_samples, n_features)
#             Samples.
#         Returns
#         -------
#         labels : array, shape [n_samples]
#             Predicted class label per sample.
#         """
#         return np.round(self.predict_proba(X))
        
#     # def __sigmoid(self, z):
#     #     """
#     #     The sigmoid function.
#     #     Parameters
#     #     ------------
#     #     z : float
#     #         linear combinations of weights and sample features
#     #         z = w_0 + w_1*x_1 + ... + w_n*x_n
#     #     Returns
#     #     ---------
#     #     Value of logistic function at z
#     #     """
#     #     return 1 / (1 + expit(-z))

#     def __sigmoid(self, x):
#         return np.array([self._sigmoid_function(value) for value in x])

#     def _sigmoid_function(self, x):
#         if x >= 0:
#             z = np.exp(-x)
#             return 1 / (1 + z)
#         else:
#             z = np.exp(x)
#             return z / (1 + z)
#     def get_params(self):
#         """
#         Get method for models coeffients and intercept.
#         Returns
#         -------
#         params : dict
#         """
#         try:
#             params = dict()
#             params['intercept'] = self.theta[0]
#             params['coef'] = self.theta[1:]
#             return params
#         except:
#             raise Exception('Fit the model first!')
            
            
class LogisticRegression():
    def __init__(self, learning_rate=0.1, regularization=None, l1_lambda=0.01, l2_lambda=0.01):
        self.losses = []
        self.train_accuracies = []
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def fit(self, x, y, epochs=150):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for i in range(epochs):
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)

    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        regularization = 0
        if self.regularization == 'l1':
            regularization = self.l1_lambda * np.abs(self.weights).sum()
        elif self.regularization == 'l2':
            regularization = self.l2_lambda * np.power(self.weights, 2).sum()
        return -np.mean(y_zero_loss + y_one_loss) + regularization

    def compute_gradients(self, x, y_true, y_pred):
        # derivative of binary cross entropy
        difference = y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])
        if self.regularization == 'l1':
            gradients_w += self.l1_lambda * np.sign(self.weights)
        elif self.regularization == 'l2':
            gradients_w += 2 * self.l2_lambda * self.weights
        return gradients_w, gradient_b
    
    # def update_model_parameters(self, error_w, error_b):
    #     if self.regularization == 'l1':
    #         self.weights = self.weights - self.learning_rate * error_w - self.learning_rate * self.lmbda * np.sign(self.weights)
    #         self.bias = self.bias - self.learning_rate * error_b
    #     elif self.regularization == 'l2':
    #         self.weights = self.weights*(1 - self.learning_rate * self.lmbda/len(y)) - self.learning_rate * error_w
    #         self.bias = self.bias - self.learning_rate * error_b
    #     else:
    #         self.weights = self.weights - self.learning_rate * error_w
    #         self.bias = self.bias - self.learning_rate * error_b
    def update_model_parameters(self, error_w, error_b):
        if self.regularization == 'l1':
            self.weights = self.weights - self.learning_rate * (error_w + self.l1_lambda * np.sign(self.weights))
            self.bias = self.bias - self.learning_rate * error_b
        elif self.regularization == 'l2':
            self.weights = self.weights - self.learning_rate * (error_w + self.l2_lambda * self.weights)
            self.bias = self.bias - self.learning_rate * error_b
        else:
            self.weights = self.weights - self.learning_rate * error_w
            self.bias = self.bias - self.learning_rate * error_b
    
    
    
    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]
    
    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])
    
    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)
    
    def _transform_x(self, x):
        x = copy.deepcopy(x)
        return x
    
    def _transform_y(self, y):
        y = copy.deepcopy(y)
        return y.reshape(y.shape[0], 1)
