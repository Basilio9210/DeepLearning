# -*- coding: utf-8 -*-
"""
Created on Sep 21 09:32:33 2018

@author: Jose Pamplona
"""

import numpy as np
import tensorflow as tf

# Se fabrican los datos a los que se les hará la regresión lineal
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 3 + 2
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.3))(y_data)

# Parametros iniciales de la ecuacion de la regresión
a = tf.Variable(-1.0)
b = tf.Variable(0.2)
y = a * x_data + b

# definición de la función de pérdida y el optimizador
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(15):     # iteraciones de optimización
        _, cost,A,B = sess.run([train,loss,a,b])
        print(step, [A,B], cost)
