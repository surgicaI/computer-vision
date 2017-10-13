from scipy import misc
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np

def convolve(image, kernel):
    # kernel = [1 2 1]/4
    rows, cols = image.shape
    kernel_width = kernel.shape[0]
    kernel_center = int((kernel_width - 1) / 2)
    # assuming valid boundary conditions
    row_convolution = kernel[kernel_center]*image[:, kernel_center:cols-kernel_center]
    for i in range(kernel_center):
        j = i + 1
        row_convolution = row_convolution + kernel[kernel_center-j]*image[:, kernel_center-j:cols-kernel_center-j] + kernel[kernel_center+j]*image[:, kernel_center+j:cols-kernel_center+j]
    image = row_convolution

    rows, cols = image.shape
    col_convolution = kernel[kernel_center]*image[kernel_center:rows-kernel_center, :]
    for i in range(kernel_center):
        j = i + 1
        col_convolution = col_convolution + kernel[kernel_center-j]*image[kernel_center-j:rows-kernel_center-j, :] + kernel[kernel_center+j]*image[kernel_center+j:rows-kernel_center+j, :]
    image = col_convolution
    # final size of the image is [image.rows - 2, image_cols -2]
    return image


if __name__ == '__main__':
    image_name = input('please input image path:')
    try:
        image = misc.imread(image_name, flatten=True)
    except:
        print('invalid image path')
        quit()

    kernel_width = 2
    while kernel_width % 2 == 0 or kernel_width < 3:
        kernel_width = int(input('please input kernel width(odd natural number >= 3):'))

    kernel = np.array([0.25, 0.5, 0.25])
    new_kernel = np.array([0.25, 0.5, 0.25])
    while(new_kernel.shape[0] < kernel_width):
        new_kernel = signal.convolve(kernel, new_kernel)
    print('input-image-shape:', image.shape)
    image = convolve(image, new_kernel)
    print('output-image-shape:', image.shape)
    plt.imshow(image, cmap='Greys_r')
    plt.show()
