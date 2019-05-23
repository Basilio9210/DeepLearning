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
