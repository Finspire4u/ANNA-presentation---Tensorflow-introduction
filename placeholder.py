"""
Linsheng He 113428893 TensorFlow Tutorial

placeholder and feed_dict
To hold nothing until you define it when use sess.run()
"""
from __future__ import print_function
import tensorflow as tf

input1 = tf.placeholder(tf.float32,[2])
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.0, 5.0], input2: [2.0,3.0]}))
