#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import tensorflow as tf

tf.random.set_seed(0)

X = np.array([1., 2., 3., 4.])
Y = np.array([1., 3., 5., 7.])

W = tf.Variable(tf.random.normal([1], -100., 100.))
# W = tf.Variable([5.0])


for step in range(300):
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    learning_rate = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))
    descent = W - tf.multiply(learning_rate, gradient)
    W.assign(descent)
    
    if step % 10 == 0:
        print("{:5} | {:10.4f} | {:10.6f}".format(step, cost.numpy(), W.numpy()[0]))


# In[ ]:




