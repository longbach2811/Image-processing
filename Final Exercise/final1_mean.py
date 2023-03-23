import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage

def Mean_Filter(img):
    mean_filter = np.array(([1/16, 2/16, 1/16],
                            [2/16, 4/16, 2/16],
                            [1/16, 2/16, 1/16]), dtype="float")
    filtered_img = ndimage.convolve(img, mean_filter)
    return filtered_img

if __name__ == "__main__":
    img = cv.imread("final1.bmp", cv.IMREAD_GRAYSCALE)

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    filtered_img = Mean_Filter(img)
    plt.subplot(1,2,2)
    plt.imshow(filtered_img, cmap='gray')
    plt.title("Mean Filtered Image")
    plt.axis("off")


    plt.show()
