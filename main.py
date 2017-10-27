from utils.io import rgb2gray, readImage
from matplotlib import pyplot
from edge.canny import canny_edge_detection
from hough.hough import hough_line
from PIL import Image


def main():
    image = readImage('im02.jpg')
    img = rgb2gray(image)
    edge = canny_edge_detection(img, 5, 1.2, 40, 90)
    accu, theta, rhos = hough_line(edge)

    print(accu)

    img_show = Image.fromarray(accu)
    img_show = img_show.convert("RGB")
    img_show.save('canny_edge.png')


if __name__ == '__main__':
    main()