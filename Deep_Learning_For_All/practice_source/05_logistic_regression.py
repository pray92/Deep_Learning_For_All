#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import tensorflow as tf

data = np.array([
                 # x1, x2, y
                 [1, 2, 0],
                 [2, 3, 0],
                 [3, 1, 0],
                 [4, 3, 1],
                 [5, 3, 1],
                 [6, 2, 1]
                 ], dtype=np.float32)

x_train = data[:,:-1]
y_train = data[:,[-1]]

x_test = np.array([[5, 2]], dtype=np.float32)
y_test = np.array([[1]], dtype=np.float32)

# tf.data.Dataset 파이프라인을 이용하여 값을 입력
# from_tensor_slices 메서드를 이용하면 list, numpy, tensorflow 자료형에서 데이터셋을 만들 수 있음
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))

W = tf.Variable(tf.random.normal([2, 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

def logistic_regression(features):
    hypothesis = tf.sigmoid(tf.matmul(features, W) + b)
    return hypothesis

def cost_function(features, labels):
    hypothesis = logistic_regression(features)
    cost = -tf.reduce_mean(labels * tf.math.log(hypothesis) + (1 - labels) * tf.math.log(1 - hypothesis))
    return cost

def grad(hypothesis, features, labels):
    with tf.GradientTape() as tape:
        cost_value = cost_function(features, labels)
    return tape.gradient(cost_value, [W, b])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

EPOCHS = 5400

for step in range(EPOCHS + 1):
    for features, labels in iter(dataset):
        hypothesis = logistic_regression(features)
        grads = grad(hypothesis, features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
        if step % 300 == 0:
            print("iter : {}, Lose : {:.4f}".format(step, cost_function(features, labels)))


# In[11]:


def accuracy_function(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy

test_acc = accuracy_function(logistic_regression(x_test), y_test)
print('Accuracy : {}%'.format(test_acc * 100))

