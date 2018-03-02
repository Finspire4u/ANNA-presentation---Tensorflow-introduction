"""
Linsheng He 113428893 TensorFlow data visualization
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
from add_layers import add_layer
import matplotlib.pyplot as plt

x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(2001):
        sess.run(train, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            #print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)
