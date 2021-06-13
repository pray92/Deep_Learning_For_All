#!/usr/bin/env python
# coding: utf-8

# In[31]:


import tensorflow as tf

x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

# 이거슨 임의(Random) 변수
W = tf.Variable(2.9)
b = tf.Variable(0.5)

# 평균
v = [1., 2., 3., 4.]
print(tf.reduce_mean(v))

# 제곱
print(tf.square(v))

learning_rate = 0.01

print("{:6}|{:10}|{:10}|{:10}".format( "Epochs", "W", "b", "cost"))
for i in range(100 + 1):
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
        W_grad, b_grad = tape.gradient(cost, [W, b])

    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
        print("{:6}|{:10.4f}|{:10.4f}|{:10.6f}".format( i, W.numpy(), b.numpy(), cost.numpy()))


