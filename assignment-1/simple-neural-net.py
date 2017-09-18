import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F

dtype = torch.DoubleTensor

# Create input and output data from
x = np.arange(-np.pi, np.pi, 0.01).reshape(-1,1)
y = np.cos(x)

# D_in is input dimension;
# H is hidden dimension;
# D_out is output dimension.
D_in, H, D_out = 1, 10, 1

# Create plot for original function
plt.scatter(x, y, s=5, marker='.', c='blue')

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# Create plot for predicted function before training
h = x.dot(w1)
h_relu = np.tanh(h)
y_pred = h_relu.dot(w2)
# plt.scatter(x, y_pred, s=5, marker='.', c='green')

# learning weights
learning_rate = 1e-6
learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)
    h_relu = np.tanh(h)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    calc = x * grad_y_pred
    temp = 1 - np.tanh(x.dot(w1))**2
    temp = temp * w2.T
    temp = temp * calc
    grad_w1 = temp.sum(axis=0).reshape(1,10)
    # print(temp.shape)
    # print(calc.shape)
    # print(w2.shape)
    # input()
    # temp = temp.dot(w2) * x
    # grad_w1 = temp * grad_y_pred
    # print(grad_w1.shape)
    # input()
    # grad_h_relu = grad_y_pred.dot(w2.T)
    # grad_h = grad_h_relu.copy()
    # grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

# Create plot for predicted function after training
h = x.dot(w1)
h_relu = np.tanh(h)
y_pred = h_relu.dot(w2)
plt.scatter(x, y_pred, s=5, marker='.', c='red')
plt.show()
