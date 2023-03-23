import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import ndimage

def Sobel_filter(img, direction):
    if (direction == 'x'):
        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        conv_val = ndimage.convolve(img, gx)
    if (direction == 'y'):
        gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        conv_val = ndimage.convolve(img, gy)
    return conv_val

def normalize(img):
    Norm_ = img/np.max(img)
    return Norm_

if __name__ == "__main__":
    img = cv.imread('final2.jpg', cv.IMREAD_GRAYSCALE)
    img = np.array(img, dtype=float)
    
    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')

    gx = Sobel_filter(img, 'x')
    gx = normalize(gx)

    gy = Sobel_filter(img, 'y')
    gy = normalize(gy)

    plt.subplot(2,2,2)
    plt.imshow(gx, cmap='gray')
    plt.title('Sobel horizontal')

    plt.subplot(2,2,3)
    plt.imshow(gy, cmap='gray')
    plt.title('Sobel vertical')

    mag = np.sqrt(gx**2 + gy**2)
    plt.subplot(2,2,4)
    plt.imshow(mag, cmap='gray')
    plt.title('Sobel Edge Detection Image')
    
    plt.show()