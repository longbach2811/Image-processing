import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img = cv.imread("final4.jpg", cv.IMREAD_GRAYSCALE)
    
    _,cvt_img  = cv.threshold(img, 15, 100, cv.THRESH_BINARY)

    edges = cv.Canny(cvt_img, 15, 100)
    

    kernel = np.ones((3,3), np.uint8)

    img_delation = cv.dilate(edges, kernel, iterations=1)
    img_erosion = cv.erode(img_delation, kernel, iterations=1)

    plt.subplot(2,3,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original image")

    plt.subplot(2,3,2)
    plt.imshow(cvt_img, cmap='gray')
    plt.title("Binary image")

    plt.subplot(2,3,3)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny")

    plt.subplot(2,3,4)
    plt.imshow(img_delation, cmap='gray')
    plt.title("Delation")

    plt.subplot(2,3,5)
    plt.imshow(img_erosion, cmap='gray')
    plt.title("Erosion")

    plt.subplot(2,3,6)
    plt.imshow(np.array(img, dtype=float) + img_erosion, cmap='gray')
    plt.title("Draw Contour")



    plt.show()


