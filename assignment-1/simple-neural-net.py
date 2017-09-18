import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F

dtype = torch.DoubleTensor

# Create input and output data from
x = torch.from_numpy(np.arange(-np.pi, np.pi, 0.01).reshape(-1,1))
y = torch.cos(x)
x = Variable(x, requires_grad=False)
y = Variable(y, requires_grad=False)

# D_in is input dimension;
# H is hidden dimension;
# D_out is output dimension.
D_in, H, D_out = 1, 10, 1

# Create plot for original function
plt.scatter(x.data.numpy(), y.data.numpy(), s=5, marker='.', c='blue')

# Randomly initialize weights
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

# Create plot for predicted function before training
h = x.mm(w1)
h_act = F.tanh(h)
y_pred = h_act.mm(w2)
plt.scatter(x.data.numpy(), y_pred.data.numpy(), s=5, marker='.', c='green')

# learning weights
learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)
    h_act = F.tanh(h)
    y_pred = h_act.mm(w2)

    # Compute loss
    loss = (y_pred - y).pow(2).sum()

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Variables with requires_grad=True.
    # After this call w1.grad and w2.grad will be Variables holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Update weights using gradient descent; w1.data and w2.data are Tensors,
    # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
    # Tensors.
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # Manually zero the gradients after updating weights
    w1.grad.data.zero_()
    w2.grad.data.zero_()

# Create plot for predicted function after training
h = x.mm(w1)
h_act = F.tanh(h)
y_pred = h_act.mm(w2)
plt.scatter(x.data.numpy(), y_pred.data.numpy(), s=5, marker='.', c='red')
plt.show()
