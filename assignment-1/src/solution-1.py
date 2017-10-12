from scipy import misc
from matplotlib import pyplot as plt

def convolve(image):
    # kernel = [1 2 1]/4
    rows, cols = image.shape
    # assuming valid boundary conditions
    # convolution of [1 2 1] / 4 kernel with the image
    row_convolution = 2*image[:, 1:cols-1]
    row_convolution = row_convolution + 1*image[:, :cols-2] + 1*image[:, 2:]
    image = 0.25 * row_convolution

    # the output of row convolution is convoluted again
    # with ([1 2 1]/4)transpose
    rows, cols = image.shape
    col_convolution = 2*image[1:rows-1, :]
    col_convolution = col_convolution + 1*image[:rows-2, :] + 1*image[2:, :]
    image = 0.25 * col_convolution
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

    # Applying multiple, successive Gaussian blurs to an image has the same effect as applying a single, larger Gaussian blur, whose radius is the square root of the sum of the squares of the blur radii that were actually applied.
    # For example, applying successive Gaussian blurs with radii of 6 and 8 gives the same results as applying a single Gaussian blur of radius 10, since  (6^2 + 8^2)^0.5 = 10.
    # initial radius of gausian kernel = sigma
    # after 2 iterations radius = (1^2 + 1^2)^0.5 signma = 2^0.5 sigma
    # after 3 iterations, radius = (2^(0.5*2) + 1^2) sigma = 3^0.5 sigma
    # after n iterations, radius = n^0.5 sigma
    # kernel_width = 2*radius + 1
    # in our case sigma = 1
    num_iterations = ((kernel_width - 1) / 2)**2
    num_iterations = int(num_iterations)
    print('num_iterations:', num_iterations)
    for _ in range(num_iterations):
        image = convolve(image)

    plt.imshow(image, cmap='Greys_r')
    plt.show()
