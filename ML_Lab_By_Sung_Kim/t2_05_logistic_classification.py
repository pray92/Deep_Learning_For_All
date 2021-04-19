#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=2))
# Use sigmoid activation for 0~1 problem
tf.model.add(tf.keras.layers.Activation('sigmoid'))
# Better result with loss function == 'binary_crossentropy'
# Adding acurracy metric to get accuracy report during training
tf.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])
tf.model.summary()

