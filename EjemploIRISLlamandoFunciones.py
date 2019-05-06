

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import FuncionesEjemploIRIS as fk



dataset = load_iris()

#Se almacena array de datos en variable
x = dataset.data 

#Se almacena array de clases en variable
y = dataset.target

#Desordenamos los datos, para que funcione correctamente el entrenamiento
x,y = shuffle(x,y)

#Consultamos cantidad de datos puede ser con shape o size
m = x.shape[0]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3)

n_hidden_1=12
n_hidden_2=7
n_input= 4
n_classes = 3

ph1,ph2,w1,w2,w3,b1,b2,b3 = fk.definirVariables(n_input,n_hidden_1,n_hidden_2,n_classes)

matriz = fk.definirMatrizClases(Y_train,dataset.target_names)
matriz_Test = fk.definirMatrizClases(Y_test,dataset.target_names)

#ph1 = tf.placeholder(tf.float32,[None,4])
#ph2 = tf.placeholder(tf.float32,[None,3])



salidaCapas = fk.multilayer_perceptron(ph1,w1,w2,w3,b1,b2,b3)

loss = fk.definirError('cross',ph2,salidaCapas)

optimizer = fk.definirOptimizer('ao')

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    
    for step in range(500):
         _,error=sess.run([train,loss],feed_dict={ph1:X_train, ph2:matriz})
         print(error)
         
    #Finaliza modelo de entrenamiento inicia predicción      
    pred = salidaCapas
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(ph2,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    acc = sess.run([accuracy],feed_dict={ph1:X_test,ph2:matriz_Test})
    print('Accuracy:',acc)
    saver.save(sess,'savemodel/model_softmax.ckpt')
np.save('x_test.npy',X_test)
np.save('y_test.npy',Y_test)
            
    
