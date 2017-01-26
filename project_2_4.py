# Load pickled data
import pickle
import os
import numpy as np
from random import randint
# TODO: Fill this in based on where you saved the training and testing data

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

'''
rgb2yuv = np.array([[0.299, 0.587, 0.114],
                    [-0.14713, -0.28886, 0.436],
                    [0.615, -0.51499, -0.10001]])
self.yuv = np.dot(self.rgb, rgb2yuv.T)
'''
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

import tensorflow as tf

EPOCHS = 20
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
  
    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    weights_layer1 = tf.Variable(tf.truncated_normal(shape = [5,5,3,6],mean = mu, stddev = sigma))
    biases_layer1 = tf.Variable(tf.zeros([6]))
    conv1 = tf.nn.conv2d(x, weights_layer1, strides=[1, 1, 1, 1], padding='VALID') + biases_layer1
    
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    weights_layer2 = tf.Variable(tf.truncated_normal(shape=[5,5,6,16], mean=mu, stddev=sigma))
    biases_layer2 = tf.Variable(tf.zeros([16]))
    conv2 = tf.nn.conv2d(conv1,weights_layer2,strides=[1,1,1,1], padding='VALID') + biases_layer2
    
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(conv2)
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    weights_fc1 = tf.Variable(tf.truncated_normal(shape = [400,120], mean=mu, stddev=sigma))
    biases_fc1 = tf.Variable(tf.zeros([120]))
    fc1 = tf.add(tf.matmul(fc1, weights_fc1), biases_fc1)
    # TODO: Activation.
    fc1 = tf.nn.tanh(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    weights_fc2 = tf.Variable(tf.truncated_normal(shape = [120,84],mean=mu, stddev=sigma ))
    biases_fc2 = tf.Variable(tf.zeros([84]))
    fc2 = tf.add(tf.matmul(fc1, weights_fc2), biases_fc2)
    # TODO: Activation.
    fc2 = tf.nn.tanh(fc2)
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    weights_out = tf.Variable(tf.truncated_normal(shape = [84,n_classes],mean=mu, stddev=sigma))
    biases_out = tf.Variable(tf.zeros([n_classes]))
    logits = tf.add(tf.matmul(fc2,weights_out), biases_out)
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)
rate = 0.001

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
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, 'lenet')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
					
