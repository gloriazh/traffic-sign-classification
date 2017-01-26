'''
The validation set is 20% of train data.
To normalize training data set, validation set and test set.
To initialize the weights as normal distribution with zero mean and sqrt(2/n) std.
This is the base code of my project get validation accuracy:0.9829 and test accuracy:0.9179
''' 
# Load pickled data
import pickle
import os
import numpy as np
from random import randint
import random
import pdb
# TODO: Fill this in based on where you saved the training and testing data

parameter="Project_2_6: 2:8Validation set; correctly normalize;scale weights to sqrt(2.0/n) for ReLu,sqrt(1/n) for tan,std=1; deform train set data" 
EPOCHS = 20
BATCH_SIZE = 128
rate = 0.001
wdr = 1e-4
dropout = 0.6
rate = 0.0007
training_file = os.path.abspath('train_2.p')
testing_file = os.path.abspath('test_2.p')

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:3]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


from sklearn.cross_validation import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

X_train_mean = np.mean(X_train,axis=0)
X_train_std = np.std(X_train,axis=0)
X_train =(X_train- X_train_mean)/X_train_std
X_validation =(X_validation- X_train_mean)/X_train_std
X_test = (X_test - X_train_mean)/X_train_std



from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

import tensorflow as tf



from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 1
#--------------------------------------------------------------------------  
    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 30x30x64.
    weights_layer1 = tf.Variable(tf.mul(tf.random_normal(shape = [3,3,3,64],mean = mu, stddev = sigma),np.sqrt(2.0/(3*3*3))))
    biases_layer1 = tf.Variable(tf.zeros([64]))
	
    weights_decay_layer1 = tf.mul(tf.nn.l2_loss(weights_layer1), wdr, name='weight_loss_layer1')
    tf.add_to_collection('losses', weights_decay_layer1)
	
    conv1 = tf.nn.conv2d(x, weights_layer1, strides=[1, 1, 1, 1], padding='VALID') + biases_layer1
    
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    conv1 = tf.nn.dropout(conv1,keep_prob)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # TODO: Layer 2: Convolutional. Output = 28x28x64.
    weights_layer2 = tf.Variable(tf.mul(tf.random_normal(shape=[3,3,64,64], mean=mu, stddev=sigma),np.sqrt(2.0/(3*3*64))))
    biases_layer2 = tf.Variable(tf.zeros([64]))
	
    weights_decay_layer2 = tf.mul(tf.nn.l2_loss(weights_layer2), wdr, name='weight_loss_layer2')
    tf.add_to_collection('losses', weights_decay_layer2)
	
    conv2 = tf.nn.conv2d(conv1,weights_layer2,strides=[1,1,1,1], padding='VALID') + biases_layer2
    
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    # TODO: Pooling. Input = 28x28x64. Output = 14x14x64.
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
#==========================================================================	
    # TODO: Layer 3: Convolutional. Input = 14x14x64. Output = 12x12x64. 
    weights_layer3 = tf.Variable(tf.mul(tf.random_normal(shape = [3,3,64,64],mean = mu, stddev = sigma),np.sqrt(2.0/(3*3*64))))
    biases_layer3 = tf.Variable(tf.zeros([64]))
	
    weights_decay_layer3 = tf.mul(tf.nn.l2_loss(weights_layer3), wdr, name='weight_loss_layer3')
    tf.add_to_collection('losses', weights_decay_layer3)
	
    conv3 = tf.nn.conv2d(conv2, weights_layer3, strides=[1, 1, 1, 1], padding='VALID') + biases_layer3
    
    # TODO: Activation.
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
       
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    # TODO: Flatten. Input = 6x6x64. Output = 2304.
    fc1 = flatten(conv3)
    # TODO: Layer 4: Fully Connected. Input = 2304. Output = 800.
    weights_fc1 = tf.Variable(tf.mul(tf.random_normal(shape = [2304,800], mean=mu, stddev=sigma),np.sqrt(1.0/2304)))
    biases_fc1 = tf.Variable(tf.zeros([800]))
	
    weights_decay_fc1 = tf.mul(tf.nn.l2_loss(weights_fc1), wdr, name='weight_loss_fc1')
    tf.add_to_collection('losses', weights_decay_fc1)
	
    fc1 = tf.add(tf.matmul(fc1, weights_fc1), biases_fc1)
    # TODO: Activation.
    fc1 = tf.nn.tanh(fc1)
    fc1 = tf.nn.dropout(fc1,keep_prob)
#?????????????????????????????????????????????????????????????????????????	
    # TODO: Layer 5: Fully Connected. Input = 800. Output = 400.
    weights_fc2 = tf.Variable(tf.mul(tf.random_normal(shape = [800,400],mean=mu, stddev=sigma ),np.sqrt(1.0/800)))
    biases_fc2 = tf.Variable(tf.zeros([400]))
	
    weights_decay_fc2 = tf.mul(tf.nn.l2_loss(weights_fc2), wdr, name='weight_loss_fc2')
    tf.add_to_collection('losses', weights_decay_fc2)
	
    fc2 = tf.add(tf.matmul(fc1, weights_fc2), biases_fc2)
    # TODO: Activation.
    fc2 = tf.nn.tanh(fc2)
    fc2 = tf.nn.dropout(fc2,keep_prob)
#**************************************************************************	
   
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&	
    # TODO: Layer 6: Fully Connected. Input = 400. Output = 43
    weights_out = tf.Variable(tf.mul(tf.random_normal(shape = [400,n_classes],mean=mu, stddev=sigma),np.sqrt(2.0/400)))
    biases_out = tf.Variable(tf.zeros([n_classes]))
	
    weights_decay_out = tf.mul(tf.nn.l2_loss(weights_out), wdr, name='weight_loss_out')
    tf.add_to_collection('losses', weights_decay_out)
	
    logits = tf.add(tf.matmul(fc2,weights_out), biases_out)
    logits = tf.nn.dropout(logits,keep_prob)
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, n_classes)


logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    num_examples = len(X_train)
    print("Training...")
    print()
    text_file = open("accuracy.txt", "a")
    text_file.write('%s \n' %parameter)
    text_file.write('lr=%s  batchsize = %s  epoch=%s\n'%(rate,BATCH_SIZE,EPOCHS))
    text_file.write("Validation Accuracy:\n")
    for i in range(EPOCHS):
	if i >= 11:
	    rate = rate*0.5
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            
	validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
	text_file.write("%s\n" %validation_accuracy)
    text_file.close()     
    saver.save(sess, 'lenet')
    print("Model saved")
	
text_file = open("accuracy.txt", "a")
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    text_file.write("Test Accuracy: %s\n" % test_accuracy)
    text_file.close()
					
