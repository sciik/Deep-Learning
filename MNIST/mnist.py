import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_backward(y, dy):
    return y*(1-y)*dy


def softmax(x):
    x_temp = x.T
    c = np.max(x_temp, axis=0)
    exp_x = np.exp(x_temp-c)
    sum_exp_x = np.sum(exp_x, axis=0)
    return (exp_x / sum_exp_x).T


def softmax_backward(y, t, batch_size):
    return (y-t) / batch_size


def forward(x, w, b):
    return np.dot(x, w) + b


def backward(dy, x, w, b):
    dx = np.dot(dy, w.T)
    dw = np.dot(x.T, dy)
    db = np.sum(dy, axis=0)

    return dx, dw, db


def cross_entropy_error(y, t):
    t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


(X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

hidden_size1 = 100
input_size = 784
output_size = 10
data_size = len(X_train)
batch_size = 500
iterator = 5000
learning_rate = 0.1

w1 = np.random.randn(input_size*hidden_size1).reshape(input_size, hidden_size1)
b1 = np.random.randn(hidden_size1)
w2 = np.random.randn(hidden_size1*output_size).reshape(hidden_size1, output_size)
b2 = np.random.randn(output_size)

for i in range(iterator):
    batch_mask = np.random.choice(data_size, batch_size)
    x = X_train[batch_mask]
    y = y_train[batch_mask]

    z1 = forward(x, w1, b1)
    a1 = sigmoid(z1)

    z2 = forward(a1, w2, b2)
    y_hat = softmax(z2)
    
    dz2 = softmax_backward(y_hat, y, batch_size)
    da1, dw2, db2 = backward(dz2, a1, w2, b2)

    dz1 = sigmoid_backward(a1, da1)
    dx, dw1, db1 = backward(dz1, x, w1, b1)

    w1 -= learning_rate*dw1
    w2 -= learning_rate*dw2
    b1 -= learning_rate*db1
    b2 -= learning_rate*db2
    
    if (batch_size*i) % data_size == 0:
        z1 = forward(X_test, w1, b1)
        a1 = sigmoid(z1)
        z2 = forward(a1, w2, b2)
        y_pre = softmax(z2)
        
        loss = cross_entropy_error(y_pre, y_test)
        y_pre = np.argmax(y_pre, axis=1)
        acc = np.sum(np.argmax(y_test, axis=1) == y_pre) / float(len(y_test))
        print("{}epoch = {}".format(int((batch_size*i)/data_size), acc))
        print("loss = {}".format(loss))
        print("====================================================")

    
    

    
    























    

