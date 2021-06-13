#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import tensorflow as tf

x_data =  [[1, 2, 1, 1],
            [2, 1, 3, 2],
            [3, 1, 3, 4],
            [4, 1, 5, 5],
            [1, 7, 5, 5],
            [1, 2, 5, 6],
            [1, 6, 6, 6],
            [1, 7, 7, 7]]

y_data =   [[0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0]]

x_data = np.asarray(x_data, dtype=np.float32)
y_data = np.asarray(y_data, dtype=np.float32)

dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data))

W = tf.Variable(tf.random.normal([4, 3], name='weight'))
b = tf.Variable(tf.random.normal([3], name='bias'))

dataset.element_spec

def cost_fn(features, labels):
    hypothesis = tf.nn.softmax(tf.matmul(features, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(labels * tf.math.log(hypothesis), axis=1))
    return cost

def grad(features, labels):
    with tf.GradientTape() as tape:
        cost_value = cost_fn(features, labels)
    return tape.gradient(cost_value, [W, b])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

EPOCHS = 2400

for step in range(EPOCHS + 1):
    for features, labels in iter(dataset):
        grads = grad(features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
        if step % 300 == 0:
            print("iter : {}, cost : {:.4f}".format(step, cost_fn(features, labels)))


# In[12]:


a = x_data
a = tf.nn.softmax(tf.matmul(a, W) + b)

print("Hypothesis : {}".format(a))

# argmax: 가장 큰 값의 index를 반환합니다.
print(tf.argmax(a, axis=1))         # 예측 값 by Hypothesis
print(tf.argmax(y_data, axis=1))    # 실제 값

