import numpy as np
import scipy.stats as st


def addPadding(image, padding):
    return np.pad(image, padding, 'constant', constant_values=0)


"""
Actual 2D gaussian formula mentioned in slides and lectures.
https://www.pixelstech.net/article/1353768112-Gaussian-Blur-Algorithm
"""
def gaussian_function(x, y, sigma):
    leftSide = 1/(2*np.pi*sigma**2)
    rightSide = np.e**(-(x**2 + y**2)/(2 * sigma**2))
    return leftSide * rightSide


"""
Creates a custom sized 2D gaussian kernel.
"""
def create_gaussian_kernel(kernel_size, sig=3):
    edge = np.floor(kernel_size/2)
    kernel = np.zeros((kernel_size,kernel_size))
    for row in range(0, kernel_size):
        for column in range(0, kernel_size):
            kernel[row][column] = \
                gaussian_function(row-edge, column-edge, sig)
    return kernel



def apply_kernel(image, kernel):
    size = len(image[1:])
    padding = np.floor(len(kernel[1:])/2)



"""
def gaussian_filter(image, kernel_size, sigma):
    kernel = create_gaussian_kernel(kernel_size, sigma)
    padding = np.floor(kernel_size/2)
    img = addPadding(image, padding)
    img = apply_kernel(img, kernel)
"""