import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def Sobel_filter(img, direction):
    if (direction == 'x'):
        gx = np.array([[-1, 0, +1], [-2, 0, 2], [-1, 0, +1]])
        conv_val = ndimage.convolve(img, gx)
    if (direction == 'y'):
        gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        conv_val = ndimage.convolve(img, gy)
    return conv_val

def normalize(img):
    Norm_ = img/np.max(img)
    return Norm_

def NMS(mag, grad):
    NMS = np.zeros(mag.shape)
    for i in range(1, int(mag.shape[0]) - 1):
        for j in range(1, int(mag.shape[1]) - 1):
            if((grad[i,j] >= -22.5 and grad[i,j] <= 22.5) or (grad[i,j] <= -157.5 and grad[i,j] >= 157.5)):
                if((mag[i,j] > mag[i,j+1]) and (mag[i,j] > mag[i,j-1])):
                    NMS[i,j] = mag[i,j]
                else:
                    NMS[i,j] = 0
            if((grad[i,j] >= 22.5 and grad[i,j] <= 67.5) or (grad[i,j] <= -112.5 and grad[i,j] >= -157.5)):
                if((mag[i,j] > mag[i+1,j+1]) and (mag[i,j] > mag[i-1,j-1])):
                    NMS[i,j] = mag[i,j]
                else:
                    NMS[i,j] = 0
            if((grad[i,j] >= 67.5 and grad[i,j] <= 112.5) or (grad[i,j] <= -67.5 and grad[i,j] >= -112.5)):
                if((mag[i,j] > mag[i+1,j]) and (mag[i,j] > mag[i-1,j])):
                    NMS[i,j] = mag[i,j]
                else:
                    NMS[i,j] = 0
            if((grad[i,j] >= 112.5 and grad[i,j] <= 157.5) or (grad[i,j] <= -22.5 and grad[i,j] >= -67.5)):
                if((mag[i,j] > mag[i+1,j-1]) and (mag[i,j] > mag[i-1,j+1])):
                    NMS[i,j] = mag[i,j]
                else:
                    NMS[i,j] = 0

    return NMS

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
    
    mag = np.sqrt(gx**2 + gy**2)
    grad = np.arctan2(gx, gy)
    plt.subplot(2,2,2)
    plt.imshow(mag, cmap='gray')
    plt.title('Magnitude')

    plt.subplot(2,2,3)
    plt.imshow(grad, cmap='gray')
    plt.title('Gradient')

    canny_img = NMS(mag, grad)
    plt.subplot(2,2,4)
    plt.imshow(canny_img, cmap='gray')
    plt.title('Non Maximum Suppression')

    plt.show()