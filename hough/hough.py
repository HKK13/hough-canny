import numpy as np
from PIL import Image, ImageDraw


def hough_peaks(H, num_peaks, threshold=0, nhood_size=5):
    """
    A function that returns the indices of the accumulator array H that
    correspond to a local maxima.  If threshold is active all values less
    than this value will be ignored, if neighborhood_size is greater than
    (1, 1) this number of indices around the maximum will be suppressed.
    """
    
    # loop through number of peaks to identify
    indices = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1) # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape) # remap to shape of H
        indices.append(H1_idx)

        # suppress indices in neighborhood
        idx_y, idx_x = H1_idx # first separate x, y indexes from argmax(H)
        # if idx_x is too close to the edges choose appropriate values
        if (idx_x - (nhood_size/2)) < 0:
            min_x = 0
        else:
            min_x = idx_x - (nhood_size//2)

        if (idx_x + (nhood_size/2) + 1) > H.shape[1]:
            max_x = H.shape[1]
        else:
            max_x = idx_x + (nhood_size//2) + 1

        # if idx_y is too close to the edges choose appropriate values
        if idx_y - (nhood_size/2) < 0:
            min_y = 0
        else:
            min_y = idx_y - (nhood_size//2)
        if idx_y + (nhood_size/2) + 1 > H.shape[0]:
            max_y = H.shape[0]
        else:
            max_y = idx_y + (nhood_size//2) + 1

        # bound each index by the neighborhood size and set all values to 0
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                # remove neighborhoods in H1
                H1[y, x] = 0

    # return the indices and the original Hough space with selected points
    return indices, H


def hough_lines_draw(img, indices, thetas, rhos):
    """
    A function that takes indices a rhos table and thetas table and draws
    lines on the input images that correspond to these values.
    """
    image = Image.fromarray(img)
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)

    for i in range(len(indices)):
        rho = rhos[indices[i][0]]
        theta = thetas[indices[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho

        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 2000*-b)
        y1 = int(y0 + 2000*a)
        x2 = int(x0 - 2000*-b)
        y2 = int(y0 - 2000*a)

        draw.line([(x1, y1), (x2, y2)], width=5, fill=(255, 0, 0))

    return np.array(image)


def hough_line(img, angle_step=1):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int((np.ceil(np.sqrt(width * width + height * height))).item())
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

    for i in range(len(x_idxs)):
        for t_idx in range(num_thetas):
            rho = int((np.round(x_idxs[i] * cos_t[t_idx] + y_idxs[i] * sin_t[t_idx])).item()) + diag_len
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos
