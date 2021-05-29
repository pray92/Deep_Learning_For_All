#!/usr/bin/env python
# coding: utf-8

# In[20]:


import tensorflow as tf
import numpy as np

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Test 용 Dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

x_data = tf.cast(x_data, dtype=np.float32)
y_data = tf.cast(y_data, dtype=np.float32)

# batch : 한번에 학습시킬 Size
dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data))

W = tf.Variable(tf.random.normal([3, 3]), name='weight')
b = tf.Variable(tf.random.normal([3]), name='bias')

def softmax_fn(features):
    hypothesis = tf.nn.softmax(tf.matmul(features, W) + b)
    return hypothesis

def cost_fn(features, labels):
    # hypothesis = softmax_fn(features)
    # cost = tf.reduce_mean(-tf.reduce_sum(labels * tf.math.log(hypothesis), 1))
    logits = tf.matmul(features, W) + b
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    
    cost = tf.reduce_mean(cost_i)

    return cost

def grad(features, labels):
    with tf.GradientTape() as tape:
        cost = cost_fn(features, labels)
        grads = tape.gradient(cost, [W, b])

    return grads

def prediction(features, labels):
    pred = tf.argmax(softmax_fn(features), 1)
    correct_pred = tf.equal(pred, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    return accuracy

# lr == 10: 최소값으로 가지않고 와리가리함
# lr == 1e-10: 이동 값이 너무 작음
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

for step in range(201):
    for features, labels in iter(dataset):
        grads = grad(features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
        cost = cost_fn(features, labels).numpy()
        acc = prediction(features, labels).numpy()
        print("iter : {}, cost : {}, accuracy : {:.4f}".format(step, cost, acc))


# In[23]:


# 테스트 데이터로 검사
x_test = tf.cast(x_test, tf.float32)
y_test = tf.cast(y_test, tf.float32)
test_cost = cost_fn(x_test, y_test).numpy()
test_acc = prediction(x_test, y_test).numpy()
print(test_cost, test_acc)

