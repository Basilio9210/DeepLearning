# -*- coding: utf-8 -*-

"""
Este código esta basado en el curso de DeepLearning del profesor Andrew Ng
"""

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v4_empty import *
from dnn_app_utils_v4_empty import load_dataset

#%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#%load_ext autoreload
#%autoreload 2

np.random.seed(1)


train_x_orig, train_y, test_x_orig, test_y = load_dataset_mnist_2classes()     # Cargar la base de datos de MNIST
#train_x_orig, train_y, test_x_orig, test_y, classes = load_data_aplication()  # Cargar la base de datos de la aplicación


# Visualizar uno de los ejemplos de la base de datos
index = 11                              # ejemplo 11
plt.figure(0)
plt.imshow(train_x_orig[index])         # visualiza ejemplo

## Obtener número de ejemplos de entrenamiento y test
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]
#
#
# imprimir características de la base de datos
print ("Número de ejemplos de entrenamiento: " + str(m_train))
print ("Número de ejemplos de testing: " + str(m_test))
print ("Tamaño de cada imagen: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))
#
#
#
# Vectorizar los ejemplos de train y test 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
##
## Normalizar los valores de las características entre 0-1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
##
print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

### Definiendo la Arquitectura de la Red ####

n_x = train_x.shape[0]     # num_px * num_px * 3
n_h = 100
n_y = train_y.shape[0]

layers_dims = (n_x, n_h, n_y)


def two_layer_model(X, YS, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implementa una red neuronal de 2-capas: INPUT->LINEAR->RELU->LINEAR->SIGMOID.
    
    Argumentos:
    X -- Datos de Entrada de tamaño (n_x, número de ejemplos)
    YS -- Vector de etiquetas deseadas (ejemplo: conteniendo 0 y 1), tamaño (1, número de ejemplos)
    layers_dims -- dimensiones de las capas (n_x, n_h, n_y)
    num_iterations -- Número de iteraciones del ciclo de optimizacióm
    learning_rate -- factor de aprendizaje de la regla del descenso del gradiente
    print_cost -- Si es True, se imprimira el costo cada 100 iteraciones
    
    Returna:
    parameters -- Diccionario que contiene W1, W2, b1, y b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                              # Almacena el costo de cada iteración
    (n_x, n_h, n_y) = layers_dims
    
    # Inicilizar el diccionario de parámetros, llamando a inicializar_Parametros
    
    ### Haga su código acá ### (≈ 1 line of code)
    
    parameters = inicializar_Parametros(n_x,n_h,n_y)

    ### Fin ###
    
    # Obtener W1, b1, W2 y b2 del diccionario de parámetros
    
    W1 = parameters["W1"]
    b2 = parameters["b2"]
    W2 = parameters["W2"]
    b3 = parameters["b3"]
    
    # Ciclo del descenso del gradiente

    for i in range(0, num_iterations):  #for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        
        ### Haga su código acá ### (≈ 2 lines of code)
        
        
        A2, cache1 = linear_activation_forward(X, W1, b2, 'relu')
        A3, cache2 = linear_activation_forward(A2, W2, b3, 'sigmoid')
        
        ### Fin ###
        
        # Calcular la función de Costo
        
        ### Haga su código acá ### (≈ 1 line of code)
        
        cost = compute_cost(A3, YS, 'LG')
        
        ### Fin ###
        
        # backward propagation: Calcular dE/dY = dE/dA3 = dA3
        
        dA3 = - (np.divide(YS, A3) - np.divide(1 - YS, 1 - A3)) # usar np.divide para derivar el costo con respecto a Al
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        
        ### Haga su código acá ###  (≈ 2 lines of code)
        
        dA2, dW2, db3 = linear_activation_backward(dA3, cache2, 'sigmoid')
        dA1, dW1, db2 = linear_activation_backward(dA2, cache1, 'relu')
        
        ### Fin ###
        
        # Configurar grads['dWl'] a dW1, grads['db1'] a db1, grads['dW2'] a dW2, grads['db2'] a db2
        
        grads['dW1'] = dW1
        grads['db2'] = db2
        grads['dW2'] = dW2
        grads['db3'] = db3
        
        # Actualizar Parámetros.
        
        ### Haga su código acá ###  (≈ 1 line of code)
        
        parameters = update_parameters(parameters,grads,learning_rate)
        
        ### gin ###

        # Obtener W1, b1, W2, b2 de parameters

        W1 = parameters["W1"]
        b2 = parameters["b2"]
        W2 = parameters["W2"]
        b3 = parameters["b3"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # gráficar el cost
    plt.figure(1)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2000, print_cost=True)

