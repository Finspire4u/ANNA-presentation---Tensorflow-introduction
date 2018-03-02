"""
Linsheng He 113428893 TensorFlow data visualization

Basic: Weights, biases, loss, Session

"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

############### create data ###################
# Generate 100 random numbers（from 0 to 1）
# Because in TensorFlow, almost all data type are float32
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

######### create tensorflow structure #########
#use random uniform to create this 1-D Variable
Weights = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
biases = tf.Variable(tf.random_uniform([1],-10.0,10.0))
# it will learn to improve initialized value to an more accurate level
y = Weights*x_data + biases
# calculate the error between y and y_data
loss = tf.reduce_mean(tf.square(y-y_data))
# create an optimizer to minimize this loss
# this is the basic optimizer and learning rate normally less than 1
optimizer = tf.train.GradientDescentOptimizer(0.5)
# run this minimization for every training step, so that loss would be smaller than last time
train = optimizer.minimize(loss)
####### create tensorflow structure end #######

# to activated all Variables
init = tf.global_variables_initializer() # important

# Session just like pointer, to point the location and run the particular area of this structure
#sess = tf.Session()
with tf.Session() as sess:
# TensorFlow will only active this structure one time when you run it
    sess.run(init) # very important & don't forget
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, 'Weights:',sess.run(Weights), 'biases:',sess.run(biases))
