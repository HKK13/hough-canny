import numpy as np


def addPadding(image, padding):
    return np.pad(image, padding, 'constant', constant_values=0)


def create_gaussian_kernel(length, sigma=3):
    """
    Creates a custom sized 2D gaussian kernel.
    """
    kernel = np.zeros((length, length))

    for y in range(length):
        for x in range(length):
            kernel_x = x - length//2
            kernel_y = y - length//2
            dist = 1/(2 * np.pi * np.power(sigma, 2))
            gau = np.power(np.e, -(np.power(kernel_x, 2) + np.power(kernel_y, 2))/(2 * np.power(sigma, 2)))
            kernel[y][x] = dist * gau

    return kernel/np.sum(kernel)


def convolve(image, kernel):
    """
    Takes an image and a kernel and returns the convolution of them
    """
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros_like(image)           # convolution output

    kernel_i, kernel_j = kernel.shape
    padding = kernel_i // 2

    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2 * padding, image.shape[1] + 2 * padding))
    image_padded[padding:-padding, padding:-padding] = image
    for x in range(image.shape[1]):         # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y:y + kernel_i, x:x + kernel_j]).sum()
    return output


def gaussian_filter(image, kernel_size, sigma=3):
    kernel = create_gaussian_kernel(kernel_size, sigma)
    return convolve(image, kernel)

