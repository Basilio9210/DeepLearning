'''
A Long - Short Term Memory Networks (LSTM) implementation using TensorFlow..
A prediction of a word after n_input words learned from text file.
A story is automatically generated if some initial words are provided to
feed the model as input. 
'''


from google.colab import drive
#drive.mount("/content/data/")
!ls data/'My Drive'/'Colab Notebooks'/'Datasets'/'LSTM_words'/

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.nn import rnn_cell
import random
import collections
import time

start_time = time.time()

# Define a log file to sum up our model
# Conveniently, the log will be stored in our data path 
data_path = "data/My Drive/Colab Notebooks/Datasets/LSTM_words/"
#writer = tf.summary.FileWriter(data_path)

# Text file containing words for training
training_file = 'belling_the_cat.txt'

# Reading text file
def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [word for i in range(len(content)) for word in content[i].split()]
    content = np.array(content)
    return content

training_data = read_data(data_path+training_file)
print("Training data loaded...")



# Define parameters
learning_rate = 0.001
n_input = 3
num_epochs = 20000
num_classes = 2
echo_step = 500
#echo_step = 1000
batch_size = 5
words_to_predict = 10


# number of units in RNN cell
n_hidden = 512


# --- Create placeholders
batchX_placeholder = tf.placeholder(tf.float32, [None, _ , 1])
batchY_placeholder = tf.placeholder(tf.float32, [None, vocab_size])

init_state = tf.placeholder(tf.float32, [batch_size, vocab_size])

# --- Weights, Bias initialization
W = tf.Variable(np.random.rand( _ , vocab_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1, _ )), dtype=tf.float32)


def model(input_placeholder, weights, biases):
    
    # reshape to [1, n_input]
    input_placeholder = tf.reshape(input_placeholder, [-1, _ ])
    
    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    input_placeholder = tf.split(input_placeholder, n_input,1)
    
    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    cell = rnn_cell.LSTMCell (n_hidden, reuse=tf.AUTO_REUSE)
    
    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    # cell = rnn.MultiRNNCell([rnn_cell.LSTMCell(n_hidden), rnn_cell.LSTMCell(n_hidden)])
    
    # generate prediction
    outputs, states = rnn.static_rnn(cell, input_placeholder, dtype=tf.float32)
    
    # there are n_input outputs but
    # we only want the last output
    return _ 

predictions = model(batchX_placeholder, W, b)


# Loss and optimizer
total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = _ , labels= _ ))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(total_loss)

# Model evaluation
# Introduce the accuracy estimation based on predictions
correct_predictions = tf.equal(tf.argmax( _ ,1), tf.argmax( _ ,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
