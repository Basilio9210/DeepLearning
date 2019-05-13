# -*- coding: utf-8 -*-
"""
Created on Mon Sep  29 10:32:33 2018

@author: Jose Pamplona
"""
from __future__ import print_function
import tensorflow as tf

# Importar la base de datos MNIST al interior de los ejemplos de TF
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parametros de entrenamiento
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Parametros de la red neuronal
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# ELABORACIÓN DEL GRAFO DE TENSORFLOW
# Definición de placeholders
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
# Declaración de los pesos y los bias (weight & bias)
weights = {'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.1)),
           'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=0.1)),
           'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes],stddev=0.1))
          }
biases = {'b1': tf.Variable(tf.truncated_normal([n_hidden_1],stddev=0.1)),
          'b2': tf.Variable(tf.truncated_normal([n_hidden_2],stddev=0.1)),
          'out': tf.Variable(tf.truncated_normal([n_classes],stddev=0.1))
         }
# Definición del perceptrón multicapa
def multilayer_perceptron(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1']) # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']) # Hidden fully connected layer with 256 neurons
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out'] # Output fully connected layer with a neuron for each class
    return out_layer
# Declarar la operación que aplica el MLP usando la información de entrada
logits = multilayer_perceptron(X)

# Declarar las operaciónes que establecen la funcion de perdida y optimización 
# para el entrenamiento.
loss_op = tf.reduce_mean(tf.losses.absolute_difference(predictions=tf.nn.softmax(logits),labels=Y))
optimizer =tf.train.GradientDescentOptimizer(learning_rate)
#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()


# Inicio de sesion para la ejecusión del Grafo
with tf.Session() as sess:
    sess.run(init)
    
    # Ciclo en el que se repetirá el proceso de entrenamiento
    for epoch in range(training_epochs):
        avg_cost = 0.
        #obtiene el numero de grupos en que queda dividida la base de datos
        total_batch = int(mnist.train.num_examples/batch_size) 
        # ciclo para entrenar con cada grupo de datos
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Correr la funcion de perdida y la operacion de optimización 
            # con la respectiva alimentación del placeholder
            _,c =sess.run([train_op, loss_op],feed_dict={X:batch_x,Y:batch_y})
            # Promedio de resultados de la funcion de pérdida
            avg_cost += c / total_batch
        # Mostrar el resultado del entrenamiento por grupos
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")


    # Prueba de los pesos y los bias obtenidos
    pred = tf.nn.softmax(logits)  # aplicando softmax al modelo logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculo de la presición
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("predictions:", correct_prediction.eval({X: mnist.test.images, Y: mnist.test.labels}))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
