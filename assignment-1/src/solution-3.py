import numpy as np
import os

def verify(X, P, x):
    threshold = 1e-5
    for i in range(X.shape[0]):
        vec = list(X[i])
        vec.append(1)
        vec = np.array(vec).reshape(4,1)
        calculated = P.dot(vec)
        calculated = calculated[:2].reshape(1, -1) / calculated[2]
        actual = x[i].reshape(1, -1)
        diff = calculated - actual
        if np.any([k>threshold for k in diff]):
            print('actual and calculated values are not equal')
            print('actual:', actual, 'calculated:', calculated)
            break
    else:
        print('actual and calculated values are equal')

def cameraProjection():
    world_file = 'res/world.txt'
    image_file = 'res/image.txt'

    # world coordinates matrix
    X = np.loadtxt(world_file).T
    # image coordinates matrix
    x = np.loadtxt(image_file).T

    A = []
    for i in range(10):
        xi, yi = x[i]
        Xi, Yi, Zi = X[i]
        a1 = [0, 0, 0, 0, -1*Xi, -1*Yi, -1*Zi, -1, yi*Xi, yi*Yi, yi*Zi, yi]
        a2 = [Xi, Yi, Zi, 1, 0, 0, 0, 0, -1*xi*Xi, -1*xi*Yi, -1*xi*Zi, -1*xi]
        A.append(a1)
        A.append(a2)

    A = np.array(A)
    if not os.path.isfile('A.txt'):
        print('writing A in A.txt')
        with open('A.txt', 'w+') as handle:
            for i in range(20):
                writeIt = list(A[i])
                writeIt = [str(k) for k in writeIt]
                handle.write(', '.join(writeIt) + '\n')

    U, s, V = np.linalg.svd(A)
    P = None
    # The value of p that minimizes Ap subject to ||p||2 = 1 is given by the eigenvector
    # corresponding to the smallest singular value of A. To find this, compute
    # the SVD of A, picking this eigenvector and reshaping it into a 3 by 4
    # matrix P.
    my_min = 2**64
    for index, val in enumerate(s):
        if val < my_min:
            my_min = val
            P = V[index]
    P = P.reshape(3, 4)

    if not os.path.isfile('P.txt'):
        print('writing P in P.txt')
        with open('P.txt', 'w+') as handle:
            for i in range(3):
                writeIt = list(P[i])
                writeIt = [str(k) for k in writeIt]
                handle.write(', '.join(writeIt) + '\n')

    # Verify your answer by re-projecting the world points X and checking
    # that they are close to x
    verify(X, P, x)


if __name__ == '__main__':
    cameraProjection()
