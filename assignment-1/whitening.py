# imports
import torch
import matplotlib.pyplot as plt
import numpy as np

def whiten():
    # Load up the 2D dataset from the file assign1 data.py.
    my_tensor = torch.load('assign0_data.py')

    # Visualize it by making a 2D scatter plot (e.g. using matplotlib)
    plt.scatter(my_tensor[:, 0].numpy(), my_tensor[:, 1].numpy(),
                s=10, marker='.', c='b')
    plt.show()
    plt.clf()

    # Translates the data so that it has zero mean (i.e. is centered at the origin).
    my_tensor -= torch.from_numpy(my_tensor.numpy().mean(axis=0))

    # plotting zero mean data
    plt.scatter(my_tensor[:, 0].numpy(), my_tensor[:, 1].numpy(),
                s=10, marker='.', c='b')
    plt.show()
    plt.clf()

    # calculating covariance
    X = my_tensor.numpy()
    cov = np.dot(X.T, X) / X.shape[0]
    # U = eigenvectors, S = eigenvalues
    U,S,V = np.linalg.svd(cov)

    # decorrelate the data
    decorrelated_data = np.dot(X, U)

    # plotting the decorrelated data
    plt.scatter(decorrelated_data[:, 0], decorrelated_data[:, 1],
                s=10, marker='.', c='b')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    whiten()
