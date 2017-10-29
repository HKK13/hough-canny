from utils.io import rgb2gray, readImage
from edge.canny import canny_edge_detection
from hough.hough import hough_line, hough_lines_draw, hough_peaks
from PIL import Image


def writeImage(index, extension, edge, accumulation, line_image):
    img_show = Image.fromarray(edge)
    img_show = img_show.convert("RGB")
    img_show.save('images/outputs/img0{0}_canny_edge.{1}'.format(index, extension))
    img_show = Image.fromarray(accumulation)
    img_show = img_show.convert("RGB")
    img_show.save('images/outputs/img0{0}_accumulation.{1}'.format(index, extension))
    img_show = Image.fromarray(line_image)
    img_show = img_show.convert("RGB")
    img_show.save('images/outputs/img0{0}_lines.{1}'.format(index, extension))


def main():
    image1 = readImage('images/im01.jpg')
    img1 = rgb2gray(image1)
    image2 = readImage('images/im02.jpg')
    img2 = rgb2gray(image2)
    image3 = readImage('images/im03.png')
    img3 = rgb2gray(image3)

    edge1 = canny_edge_detection(img1, 5, 1.4, 10, 70)
    accu1, thetas1, rhos1 = hough_line(edge1)
    indices1, accu1 = hough_peaks(accu1, 30, 100, 11)

    edge2 = canny_edge_detection(img2, 5, 3, 30, 60)
    accu2, thetas2, rhos2 = hough_line(edge2)
    indices2, accu2 = hough_peaks(accu2, 10, 200, 11)

    edge3 = canny_edge_detection(img3, 5, 2, 30, 70)
    accu3, thetas3, rhos3 = hough_line(edge3)
    indices3, accu3 = hough_peaks(accu3, 25, 500, 11)

    drawed_lines1 = hough_lines_draw(img1, indices1, thetas1, rhos1)
    drawed_lines2 = hough_lines_draw(img2, indices2, thetas2, rhos2)
    drawed_lines3 = hough_lines_draw(img3, indices3, thetas3, rhos3)

    writeImage(1, 'jpg', edge1, accu1, drawed_lines1)
    writeImage(2, 'jpg', edge2, accu2, drawed_lines2)
    writeImage(3, 'png', edge3, accu3, drawed_lines3)


if __name__ == '__main__':
    main()