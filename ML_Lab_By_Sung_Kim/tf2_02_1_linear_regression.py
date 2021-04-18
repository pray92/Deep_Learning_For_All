import numpy as np
import tensorflow as tf

def linear_regression(X):
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]

    tf.model = tf.keras.Sequential()
    # units == output shape, input_dim == input shape
    tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

    sgd = tf.keras.optimizers.SGD(lr=0.1)  # SGD == standard gradient descendent, lr == learning rate
    tf.model.compile(loss='mse', optimizer=sgd)  # mse == mean_squared_error, 1/m * sig (y'-y)^2

    # prints summary of the model to the terminal
    tf.model.summary()

    # fit() executes training
    tf.model.fit(x_train, y_train, epochs=200)

    # predict() returns predicted value
    return tf.model.predict(np.array(X)) # y_predict

if __name__ == "__main__":
    print(linear_regression([5, 4]))