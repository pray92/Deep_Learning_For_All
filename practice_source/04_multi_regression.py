#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import tensorflow as tf

tf.random.set_seed(0)

'''
x1 = [73, 93., 89., 96., 73.,]
x2 = [80., 88., 91., 98., 66.,]
x3 = [75., 93., 90., 100., 70.,]
Y = [152., 185., 180., 196., 142.]
'''

data = np.array([
    # x1, x2, x3, Y
    [ 73., 80., 75., 152.],
    [ 93., 88., 93., 185.],
    [ 89., 91., 90., 180.],
    [ 96., 98., 100., 196.],
    [ 73., 66., 70., 142.]
], dtype=np.float32)

X = data[:, :-1]
Y = data[:, [-1]]

W = tf.Variable(tf.random.normal([3, 1]))
b = tf.Variable(tf.random.normal([1]))

learning_rate = 0.000001

# 가설 함수
def predict(X):
    return tf.matmul(X, W) + b

epochs = 2000

print("Before W", W.numpy())

for i in range(epochs + 1):
    # 비용 함수의 경사를 기록
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(X) - Y)))
    
    # 비용(손실)의 경사를 연산
    W_grad, b_grad = tape.gradient(cost, [W, b])
    
    # W와 b 업데이트
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    
    if i % 100 == 0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))
    
print("After W", W.numpy())
    

