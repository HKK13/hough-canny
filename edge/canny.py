import numpy as np


def non_max_suppression(gradient, theta):
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

            if tq == 0:  # 0 is E-W (horizontal)
                if gradient[i, j] <= gradient[i, j - 1] or gradient[i, j] <= gradient[i, j + 1]:
                    grad_sup[i, j] = 0
            if tq == 1:  # 1 is NE-SW
                if gradient[i, j] <= gradient[i - 1, j + 1] or gradient[i, j] <= gradient[i + 1, j - 1]:
                    grad_sup[i, j] = 0
            if tq == 2:  # 2 is N-S (vertical)
                if gradient[i, j] <= gradient[i - 1, j] or gradient[i, j] <= gradient[i + 1, j]:
                    grad_sup[i, j] = 0
            if tq == 3:  # 3 is NW-SE
                if gradient[i, j] <= gradient[i - 1, j - 1] or gradient[i, j] <= gradient[i + 1, j + 1]:
                    grad_sup[i, j] = 0
    return grad_sup
