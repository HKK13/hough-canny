from PIL import Image
import numpy as np


"""
Reads image into a numpy matrix.
"""
def readImage(path):
    img = Image.open(path)
    return np.array(img)

"""
Converts image into given extension format.
"""
def formatImage(img):
    return Image.fromarray(img)

"""
Converts image to gray scale.
"""
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.587, 0.114])