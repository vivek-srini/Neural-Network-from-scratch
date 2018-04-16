# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 13:17:42 2018

@author: Admin
"""

import numpy as np
import pandas as pd
X_train=pd.read_csv('Churn_Modelling.csv')
X_new=np.transpose(X_train)
X=X_new.iloc[6:-1,:].astype(float)
Y_new=X_train.iloc[:,-1].astype(float).reshape(-1,1)
Y=Y_new.T
def layer_sizes(X,Y):
    n_x=len(X)
    n_h=4
    n_y=len(Y)
    return (n_x,n_h,n_y)

def initialize_parameters(n_x,n_h,n_y):
    W1=np.random.randn(n_h,n_x)
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)
    b2=np.zeros((n_y,1))
    parameters={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
    return parameters

def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))

def forward_prop(parameters,X):
    W1=parameters['W1']
    W2=parameters['W2']
    b1=parameters['b1']
    b2=parameters['b2']
    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1)
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2)
    cache={'Z1':Z1,'A1':A1,'Z2':Z2,'A2':A2}
    return A2,cache
    


def back_prop(parameters,cache,X,Y):
    m=X.shape[1]
    W2=parameters['W2']
    A1=cache['A1']
    A2=cache['A2']
    dZ2=A2-Y
    dW2=(1 / m) * np.dot(dZ2, A1.T)
    db2=np.sum(dZ2,axis=1,keepdims=True)/m
    dZ1=np.multiply(np.dot(np.transpose(W2),dZ2),1-np.power(A1,2))
    dW1=np.dot(dZ1,np.transpose(X))
    db1=np.sum(dZ1,axis=1,keepdims=True)/m
    grads={'dW1':dW1,'dW2':dW2,'db1':db1,'db2':db2}
    return grads

def compute_cost(A2,Y):
    m=X.shape[1]
    logprobs=np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),1-Y)
    cost = -np.sum(logprobs)/m
    return cost

def Grad_Desc(parameters,grads,learning_rate=1.2):
    W1=parameters['W1']
    W2=parameters['W2']
    b1=parameters['b1']
    b2=parameters['b2']
    dW1=grads['dW1']
    dW2=grads['dW2']
    db1=grads['db1']
    db2=grads['db2']
    W1=W1-learning_rate*dW1
    W2=W2-learning_rate*dW2
    b1=b1-learning_rate*db1
    b2-b2-learning_rate*db2
    parameters={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
    return parameters

def nn_model(X,Y,n_h,num_iter):
    n_x=layer_sizes(X,Y)[0]
    n_y=layer_sizes(X,Y)[2]
    parameters=initialize_parameters(n_x,n_h,n_y)
    for i in range(num_iter):
        A2,cache=forward_prop(parameters,X)
        cost=compute_cost(A2,Y)
        grads=back_prop(parameters,cache,X,Y)
        parameters=Grad_Desc(parameters,grads)
        if i%1000==0:
            print(cost)
    return parameters
        
            
def predict(X,parameters):
    A2,cache=forward_prop(parameters,X)
    predictions=(A2>0.5)
    return predictions
parameters=nn_model(X,Y,4,10000)
prediction=predict(X,parameters)




