#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import tensorflow as tf

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]

y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

tf.model = tf.keras.Sequential()

# input_dim=3 gives multi-variable regression
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=3))
# This line can be omitted, as linear activation is default
tf.model.add(tf.keras.layers.Activation('linear'))
# Advanced reading https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d

tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5))
tf.model.summary()
history = tf.model.fit(x_data, y_data, epochs=100)

y_predict = tf.model.predict(np.array([[72., 93., 90.]]))
print(y_predict)

