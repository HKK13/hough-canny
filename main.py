from utils.io import rgb2gray, readImage
from matplotlib import pyplot
from edge.canny import canny_edge_detection
from PIL import Image


def main():
    image = readImage('im02.jpg')
    img = rgb2gray(image)
    edge = canny_edge_detection(img, 5, 3, 90, 150)

    img_show = Image.fromarray(edge)
    img_show = img_show.convert("L")
    img_show.save('canny_edge.png')


if __name__ == '__main__':
    main()