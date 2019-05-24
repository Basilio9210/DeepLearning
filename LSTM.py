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

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_list = 0

    #writer.add_graph(session.graph)

    # while step < training_iters:
    for epoch_idx in range(num_epochs): #
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)
        
        # Define the input words per batch
        symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
        
        # Define the label of words per batch
        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])
        
        # Feed the graph
        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, total_loss, predictions], \
                                                feed_dict={batchX_placeholder: _ , \
                                                           batchY_placeholder: _ })
        loss_list += loss
        acc_total += acc
        if (epoch_idx+1) % echo_step == 0:
            
            print("Step = " + str(epoch_idx+1) + ", Loss = " + \
                  "{:.6f}".format(loss_list/echo_step) + ", Accuracy= " + \
                  "{:.2f}%".format(100 * acc_total / echo_step))
            acc_total = 0
            loss_list = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
        #step += 1
        offset += (n_input+1)
    
    print("Optimization Finished!")
    
    
    flag = True
    while flag == True:
        prompt = "Write %s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        
        if words[0] == '1':
            flag = False
            break
        
        if len(words) != n_input:
            print ("Wrong num of words")
            continue
        try:
        #if True:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(words_to_predict):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(predictions, feed_dict={batchX_placeholder: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
            
        except:
        #else:
            print("Word not in dictionary")
