from scipy import misc
from matplotlib import pyplot as plt

def convolve(image):
    # kernel = [1 2 1]/4
    rows, cols = image.shape
    # assuming valid boundary conditions
    row_convolution = 2*image[:, 1:cols-1]
    row_convolution = row_convolution + 1*image[:, :cols-2] + 1*image[:, 2:]
    image = 0.25 * row_convolution

    rows, cols = image.shape
    col_convolution = 2*image[1:rows-1, :]
    col_convolution = col_convolution + 1*image[:rows-2, :] + 1*image[2:, :]
    image = 0.25 * col_convolution
    return image


if __name__ == '__main__':
    image_name = input('please input image path:')
    try:
        image = misc.imread(image_name, flatten=True)
    except:
        print('invalid image path')
        quit()

    kernel_width = 2
    while kernel_width % 2 == 0 or kernel_width <= 0:
        kernel_width = int(input('please input kernel width(odd natural number):'))

    # - So can smooth with small-width kernel, repeat, and get
    # same result as larger-width kernel would have.
    # - Convolving two times with Gaussian kernel of width σ is
    # same as convolving once with kernel of width σ√2
    num_iterations = 10
    for _ in range(num_iterations):
        image = convolve(image)

    plt.imshow(image, cmap='Greys_r')
    plt.show()
