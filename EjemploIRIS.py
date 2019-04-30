# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 18:29:50 2019

@author: Karol Ortega
"""

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#Se carga base de datos
dataset = load_iris()

#Se almacena array de datos en variable
x = dataset.data 

#Se almacena array de clases en variable
y = dataset.target

#Desordenamos los datos, para que funcione correctamente el entrenamiento
x,y = shuffle(x,y)

#Consultamos cantidad de datos puede ser con shape o size
m = x.shape[0]
#m = y.size

#x_train = x[0:m*0.7]
#x_test = x[m*0.7:]
#y_train = y[0:m*0.7]
#y_test = y[m*0.7:]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
#return (x_train,x_test ,y_train,y_test)
#
#Inicializamos las variables para los pesos
w1 = tf.get_variable("w1",[4,12],initializer=tf.contrib.layers.xavier_initializer(seed=0))
w2 = tf.get_variable("w2",[12,7],initializer=tf.contrib.layers.xavier_initializer(seed=0))
w3 = tf.get_variable("w3",[7,3],initializer=tf.contrib.layers.xavier_initializer(seed=0))

#Inicializamos las variableas para el bias
b1 = tf.get_variable("b1",[12],initializer=tf.contrib.layers.xavier_initializer(seed=0))
b2 = tf.get_variable("b2",[7],initializer=tf.contrib.layers.xavier_initializer(seed=0))
b3 = tf.get_variable("b3",[3],initializer=tf.contrib.layers.xavier_initializer(seed=0))

#Variables 
ph1 = tf.placeholder(tf.float32,[None,4])
ph2 = tf.placeholder(tf.float32,[None,3])

#Definimos matriz de y filas y 3 columnas
matriz = np.zeros((Y_train.size,dataset.target_names.size ))

#Creamos matriz de clases
for i in range(Y_train.size):
    matriz[i,Y_train[i]]=1


matriz_Test = np.zeros((Y_test.size,dataset.target_names.size ))

#Creamos matriz de clases
for i in range(Y_test.size):
    matriz_Test[i,Y_test[i]]=1
#Inicia MLP
#operaciones para una neurona    
#operacion1 = tf.matmul(ph1,w1)
#operacion2 = tf.add(operacion1,b1)        
#operacion3 = tf.nn.relu(operacion2)

#Refatorizando operaciones para una neurona
capa1=tf.nn.relu(tf.add(tf.matmul(ph1,w1),b1))
capa2=tf.nn.relu(tf.add(tf.matmul(capa1,w2),b2))
capa3=tf.nn.relu(tf.add(tf.matmul(capa2,w3),b3))

#Finaliza MLP

#Inicia Modelo de entrenamiento
#loss = tf.losses.mean_squared_error(labels = ph2,predictions = capa3)
#loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = ph2,logits = capa3)
loss = tf.losses.softmax_cross_entropy(onehot_labels = ph2,logits = capa3)

#optimizer = tf.train.GradientDescentOptimizer(0.01)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta2=0.5)
#optimizer = tf.train.MomentumOptimizer(0.01,0.9)
#optimizer = tf.train.AdadeltaOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    
    for step in range(500):
         _,error=sess.run([train,loss],feed_dict={ph1:X_train, ph2:matriz})
         print(error)
                     
#Finaliza modelo de entrenamiento inicia predicción      
    pred = capa3
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(ph2,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    acc = sess.run([accuracy],feed_dict={ph1:X_test,ph2:matriz_Test})
    print('Accuracy:',acc)
    saver.save(sess,'savemodel/model_softmax.ckpt')


