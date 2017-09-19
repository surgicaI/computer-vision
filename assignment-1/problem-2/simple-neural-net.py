import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 1, 10, 1

# Create input and output data from
x = Variable(torch.from_numpy(np.arange(-np.pi, np.pi, 0.01).reshape(-1,1)).type(torch.FloatTensor))
y = Variable(torch.cos(x.data), requires_grad=False)

# Create plot for original function
plt.scatter(x.data.numpy(), y.data.numpy(), s=5, marker='.', c='blue', label="Original cosine function")

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=False)

y_pred = model(x)
# Create plot for Y-pred before training
plt.scatter(x.data.numpy(), y_pred.data.numpy(), s=5, marker='.', c='green', label='y-pred before training')

learning_rate = 1e-4
for t in range(5000):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Variable of input data to the Module and it produces
    # a Variable of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Variables containing the predicted and true
    # values of y, and the loss function returns a Variable containing the
    # loss.
    loss = loss_fn(y_pred, y)

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Variables with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Variable, so
    # we can access its data and gradients like we did before.
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data

# Create plot for predicted function after training
y_pred = model(x)
plt.scatter(x.data.numpy(), y_pred.data.numpy(), s=5, marker='.', c='red', label='y-pred after training')
plt.legend(loc='best')
plt.show()
