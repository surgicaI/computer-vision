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

if __name__ == '__main__':
    whiten()
