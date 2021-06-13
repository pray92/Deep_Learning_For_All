#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 드라이브에 접근할 수 있도록 아래 코드 입력
from google.colab import drive
drive.mount('/content/drive')


# In[1]:


import numpy as np
import tensorflow as tf

filename ='/content/drive/MyDrive/Colab_Notebook/Deep_Learning_For_All/datasets/data-04-zoo.csv'

data = np.loadtxt(filename, delimiter=',', dtype=np.float32)

x_data = data[:, :-1]

nb_clasess = 7 # 0 ~6

y_data = data[:, [-1]]
y_data = tf.one_hot(y_data, depth=nb_clasess)
y_data = tf.reshape(y_data, [-1, nb_clasess])

dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(len(x_data))

W = tf.Variable(tf.random.normal([16, nb_clasess]), name='weight')
b = tf.Variable(tf.random.normal([nb_clasess]), name='bias')

dataset.element_spec

def cost_fn(features, labels):
    logits = tf.matmul(features, W) + b
    # hypothesis = tf.nn.softmax(logits)
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    cost = tf.reduce_mean(cost_i)

    return cost

def grad(features, labels):
    with tf.GradientTape() as tape:
        cost_value = cost_fn(features, labels)
        grads = tape.gradient(cost_value, [W, b])
    return grads

def prediction(features, labels):
    pred = tf.argmax(tf.nn.softmax(tf.matmul(features, W) + b), 1)
    correct_pred = tf.equal(pred, tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    return accuracy

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

EPOCHS = 2400

for step in range(EPOCHS + 1):
    for features, labels in iter(dataset):
        grads = grad(features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
        if step % 300 == 0:
            acc = prediction(features, labels).numpy()
            cost = cost_fn(features, labels).numpy()
            print("iter : {}, cost : {:.4f}, accuracy : {:.4f}".format(step, cost, acc))

