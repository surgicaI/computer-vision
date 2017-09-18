import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

def cosine(theta1, theta2, step_size):
    X = np.arange(theta1, theta2, step_size)
    return X, np.cos(X)

def plot_data(data, color):
    plt.scatter(data[0], data[1], s=5, marker='.', c=color)

def show_plot():
    plt.show()

class NeuralNetwork():
    def __init__(self):
        dtype = torch.FloatTensor
        D_in, H, D_out = 1, 10, 1
        self.w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
        self.w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)
        self.learning_rate = 1e-6

    def forward(self, X):
        z2 = X.mm(self.w1).clamp(min=0)
        a2 = Variable(torch.from_numpy(np.tanh(z2.data.numpy())), requires_grad=True)
        y_pred = a2.mm(self.w2)
        return y_pred

    def get_loss(self, y_pred, Y):
        return np.square(y_pred - y).sum()

if __name__ == '__main__':
    plot_data(cosine(-np.pi, np.pi, 0.01), 'blue')
    show_plot()
