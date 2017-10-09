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
    # D = Diagnal matrix of eigen values
    U,S,V = np.linalg.svd(cov)
    D = np.diag(np.power(S, -0.5))

    # decorrelate the data
    decorrelated_data = np.dot(U, D)
    decorrelated_data = np.dot(decorrelated_data, U.T)
    decorrelated_data = np.dot(X, decorrelated_data)

    # plotting the decorrelated data
    plt.scatter(decorrelated_data[:, 0], decorrelated_data[:, 1],
                s=10, marker='.', c='b')
    plt.show()
    plt.clf()

    new_cov = np.dot(decorrelated_data.T, decorrelated_data) / decorrelated_data.shape[0]
    print('covariance_of_decorrelated_data:')
    print(new_cov)

if __name__ == '__main__':
    whiten()

# For the whitened data the covariance matrix is 2X2 identity matrix.
# The X_i_j value in the covariance matrix represents the
# covariance between ith and jth vector.
# Since our data is in two dimensions, say X and Y. The covariance between X and Y
# data points is zero. Which means there is 0 dependence between these
# two data points for the whitened data.
# The variance for both X data points as well as Y data points is 1. 
