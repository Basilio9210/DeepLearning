# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:43:16 2019

@author: Karol Ortega
"""

import numpy as np
import tensorflow as tf


def definirVariables(n_input,n_hidden_1,n_hidden_2,n_classes):
    #Inicializamos las variables para los pesos
    w1 = tf.get_variable("w1",[n_input,n_hidden_1],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    w2 = tf.get_variable("w2",[n_hidden_1,n_hidden_2],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    w3 = tf.get_variable("w3",[n_hidden_2,n_classes],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    
    #Inicializamos las variableas para el bias
    b1 = tf.get_variable("b1",[n_hidden_1],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b2 = tf.get_variable("b2",[n_hidden_2],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b3 = tf.get_variable("b3",[n_classes],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    
    X = tf.placeholder(tf.float32,[None,n_input])
    Y = tf.placeholder(tf.float32,[None,n_classes])
    
    return X,Y,w1,w2,w3,b1,b2,b3
    

def multilayer_perceptron(ph1,w1,w2,w3,b1,b2,b3):
    capa1=tf.nn.relu(tf.add(tf.matmul(ph1,w1),b1))
    capa2=tf.nn.relu(tf.add(tf.matmul(capa1,w2),b2))
    capa3=tf.nn.relu(tf.add(tf.matmul(capa2,w3),b3))
    
    return capa3

def definirMatrizClases(Y_train,labels):
    #Definimos matriz de y filas y 3 columnas
    matriz = np.zeros((Y_train.size,labels.size ))
    
    #Creamos matriz de clases
    for i in range(Y_train.size):
        matriz[i,Y_train[i]]=1
        
    return matriz

def definirError(funcionError,ph2,salidaCapa):
    tipoFuncionError={
                'ms':0,
                'sgd':1,
                'cross':3
            }
    if funcionError in tipoFuncionError:        
        if funcionError == 'ms':
            loss = tf.losses.mean_squared_error(labels = ph2,predictions = salidaCapa)
        elif funcionError == 'sgd':
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = ph2,logits = salidaCapa)
        elif funcionError == 'cross':
            loss = tf.losses.softmax_cross_entropy(onehot_labels = ph2,logits = salidaCapa)
    else:
        loss = tf.losses.mean_squared_error(labels = ph2,predictions = salidaCapa)
        
    return loss

def definirOptimizer(funcionOptimizer):
    if funcionOptimizer == 'gdt':
        optimizer = tf.train.GradientDescentOptimizer(0.01)
    elif funcionOptimizer == 'ao':    
        optimizer = tf.train.AdamOptimizer()
    elif funcionOptimizer == 'mo':
        optimizer = tf.train.MomentumOptimizer(0.01,0.9)
    elif funcionOptimizer == 'adto':
        optimizer = tf.train.AdadeltaOptimizer(0.01)
    
    return optimizer