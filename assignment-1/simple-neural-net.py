import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

dtype = torch.DoubleTensor

# Create input and output data from
x = torch.from_numpy(np.arange(-np.pi, np.pi, 0.01).reshape(-1,1))
y = np.cos(x)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = x.shape[0], 1, 10, 1

# Create plot for original function
plt.scatter(x.numpy(), y.numpy(), s=5, marker='.', c='blue')

# Randomly initialize weights
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

# Create plot for predicted function before training
h = x.mm(w1)
h_act = np.tanh(h)
y_pred = h_act.mm(w2)
plt.scatter(x.numpy(), y_pred.numpy(), s=5, marker='.', c='green')

# learning weights
learning_rate = 1e-6
for t in range(10000):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_act = np.tanh(h)
    y_pred = h_act.mm(w2)

    # Compute loss
    loss = (y_pred - y).pow(2).sum()

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_act.t().mm(grad_y_pred)
    grad_h_act = grad_y_pred.mm(w2.t())
    grad_h = grad_h_act.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

# Create plot for predicted function after training
h = x.mm(w1)
h_act = np.tanh(h)
y_pred = h_act.mm(w2)
plt.scatter(x.numpy(), y_pred.numpy(), s=5, marker='.', c='red')
plt.show()
