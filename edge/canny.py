import numpy as np
from utils.filters import gaussian_filter, intensity_gradient


def non_max_suppression(gradient, theta):
    """
    Compares the current pixel with every neighbouring pixel
    to check whether if it is local maximum or not,
    if not then it is suppressed.
    """

    grad_sup = gradient.copy()

    # Quantizes the direction values between [0, 5)
    # Other way is to look for angles in conditionals
    # 0-22.5 or 157.5-202.5 or 337.5-360    => 0 degrees
    # 22.5-67.5 or 202.5-247.5              => 45 degrees
    # 67.5-112.5 or 247.5 to 292.5          => 90 degrees
    # 112.5-157.5 or 292.5-337.5            => 135 degrees
    theta_q = (np.round(theta * (5.0 / np.pi)) + 5) % 5

    for i in range(gradient.shape[0]):
        for j in range(gradient.shape[1]):

            # Suppress pixels at the image edge
            if i == 0 or i == gradient.shape[0] - 1 or j == 0 or j == gradient.shape[1] - 1:
                grad_sup[i, j] = 0
                continue

            tq = theta_q[i, j] % 4

            if tq == 0:  # E-W
                if gradient[i, j] <= gradient[i, j - 1] or gradient[i, j] <= gradient[i, j + 1]:
                    grad_sup[i, j] = 0
            if tq == 1:  # NE-SW
                if gradient[i, j] <= gradient[i - 1, j + 1] or gradient[i, j] <= gradient[i + 1, j - 1]:
                    grad_sup[i, j] = 0
            if tq == 2:  # 2 N-S
                if gradient[i, j] <= gradient[i - 1, j] or gradient[i, j] <= gradient[i + 1, j]:
                    grad_sup[i, j] = 0
            if tq == 3:  # NW-SE
                if gradient[i, j] <= gradient[i - 1, j - 1] or gradient[i, j] <= gradient[i + 1, j + 1]:
                    grad_sup[i, j] = 0
    return grad_sup


def standard_threshold(gradient, threshold):
    return gradient > threshold


def hysteresis_threshold(gradient, low_threshold, high_threshold):
    """
    Applies two thresholds for deciding if the edges are really edges.
    Ones higher than the high_threshold considered strong edges and the
    ones that are lower than low_threshold set to 0. Finally values between
    are checked if they are connected to strong edges.
    """

    strong_edges = (np.array(gradient) > high_threshold)
    edges = (np.array(gradient) > low_threshold)
    output = strong_edges.copy()

    height, width = gradient.shape

    current_pixels = []
    for i in range(1, height-1):
        for j in range(1, width-1):

            if edges[i, j] != 1:
                continue

            local_patch = edges[i-1:i+2, j-1:j+2]
            patch_max = local_patch.max()

            if patch_max == 2:
                current_pixels.append((i, j))
                output[i, j] = 1

    temp_pixels = []
    while len(current_pixels) > 0:
        for i, j in current_pixels:
            for di in range(-1, 2):
                for dj in range(-1, 2):

                    if di == 0 and dj == 0:
                        continue

                    i2 = i + di
                    j2 = j + dj

                    if edges[i2, j2] == 1 and output[i2, j2] == 0:
                        temp_pixels.append((i2, j2))
                        output[i2, j2]
        current_pixels = temp_pixels

    return output


def canny_edge_detection(image, kernel_size, sigma, low_threshold, high_threshold):
    gaussian_blurred = gaussian_filter(image, kernel_size, sigma)
    gradient, theta = intensity_gradient(gaussian_blurred)
    suppressed_gradient = non_max_suppression(gradient, theta)
    thresholded_image = [[]]

    if not high_threshold:
        thresholded_image = standard_threshold(suppressed_gradient, low_threshold)
    else:
        thresholded_image = hysteresis_threshold(suppressed_gradient, low_threshold, high_threshold)

    return thresholded_image
