from utils.io import rgb2gray, readImage
from matplotlib import pyplot
from utils.filters import \
    addPadding, create_gaussian_kernel, gaussian_function


def main():
    kernel = create_gaussian_kernel(5, 1.5)
    print(kernel)

if __name__ == '__main__':
    main()