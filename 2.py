from __future__ import print_function
import tensorflow as tf
import numpy as np
from add_layers import add_layer

x_data = np.random.rand(100).astype(np.float32)
y_data = tf.square(x_data) + 0.3

l1 = add_layer(x_data, 1, 10, activation_function = tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function = None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction),reduction_indices = [1]))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(2001):
        sess.run(train)
        if i % 50 == 0:
            print(sess.run(train))
