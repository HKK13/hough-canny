import numpy as np


def hough_line(img, angle_steps=1):
    thetas = np.deg2rad(np.arange(-90.0, 90.0), angle_steps)    # Spacing between angles.
    width, height = img.shape
    diag_len = int((np.ceil(np.sqrt(width * width + height * height))).item())   # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

    # Values for a line in Polar coordinate system: ρ = x cos θ + y sin θ
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        for j in range(num_thetas):
            # ρ = (x cos θ + y sin θ)
            rho_f = round(x_idxs[i] * cos_t[j] + y_idxs[i] * sin_t[j]) + diag_len
            rho_index = int(rho_f.item())
            theta_index = j
            accumulator[rho_index, theta_index] += 1

    return accumulator, thetas, rhos
