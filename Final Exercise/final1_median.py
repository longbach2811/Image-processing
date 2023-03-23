import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage

def Median_filter(img):
    r, c = img.shape
    img_temp = np.zeros([r, c])
    for i in range(1, r - 1):
        for j in range(1, c - 1):
            temp = np.array([img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1],
                            img[i, j - 1], img[i, j], img[i, j + 1],
                            img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1]], dtype=float)

            temp = sorted(temp)
            img_temp[i, j] = np.median(temp)
    return img_temp
                 

if __name__ == "__main__":
    img = cv.imread("final1.bmp", cv.IMREAD_GRAYSCALE)
    img = np.array(img, dtype=float)
    
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    img1 = Median_filter(img)
    plt.imshow(img1, cmap='gray')
    plt.title("Median Filtered Image")
    plt.axis("off")

    plt.show()