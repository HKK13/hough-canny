import numpy as np
from scipy import ndimage


def addPadding(image, padding):
    return np.pad(image, padding, 'constant', constant_values=0)


"""
Creates a custom sized 2D gaussian kernel.
"""
def create_gaussian_kernel(length, sigma=3):
    kernel = np.zeros((length, length))

    for y in range(length):
        for x in range(length):
            kernel_x = x - length//2
            kernel_y = y - length//2
            dist = 1/(2 * np.pi * np.power(sigma, 2))
            gau = np.power(np.e, -(np.power(kernel_x, 2) + np.power(kernel_y, 2))/(2 * np.power(sigma, 2)))
            kernel[y][x] = dist * gau

    return kernel/np.sum(kernel)


def gaussian_filter(image, kernel_size, sigma=3):
    kernel = create_gaussian_kernel(kernel_size, sigma)
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape

    filtered = np.zeros_like(image)

    for y in range(i_height):
        for x in range(i_width):
            weighted_pixel_sum = 0

            for ky in range(-(k_height // 2), k_height//2):
                for kx in range(-(k_width // 2), k_width//2):
                    pixel = 0
                    pixel_y = y - ky
                    pixel_x = x - kx

                    if (pixel_y >= 0) and (pixel_y < i_height) and (pixel_x >= 0) and (pixel_x < i_width):
                        pixel = image[pixel_y][pixel_x]

                    # get the weight at the current kernel position
                    weight = kernel[ky + (k_height // 2)][kx + (k_width // 2)]

                    # weigh the pixel value and sum
                    weighted_pixel_sum += pixel * weight

            # finally, the pixel at location (x,y) is the sum of the weighed neighborhood
            filtered[y][x] = weighted_pixel_sum
    return filtered

