
import matplotlib.pyplot as plt
import numpy as np
import cv2
#loading image
image = cv2.imread('Task2.png')

# to show image
plt.figure(figsize=(10,10))
plt.imshow(image, cmap = 'gray')
plt.show()


#converting to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Smooth the image with a Gaussian filter
gaussian = cv2.GaussianBlur(gray,(3,3),0)

#Sobel Edge Detector
sobelx = cv2.Sobel(gaussian,cv2.CV_8U,1,0,ksize=5)
sobely = cv2.Sobel(gaussian,cv2.CV_8U,0,1,ksize=5)
sobel = sobelx + sobely

#Prewitt Gradient Operator
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewittx = cv2.filter2D(gaussian, -1, kernelx)
prewitty = cv2.filter2D(gaussian, -1, kernely)

#Laplacian of Gaussian
laplacian = cv2.Laplacian(image,cv2.CV_64F)

#The Canny operator
canny = cv2.Canny(image,100,200)

#showing results


cv2.imshow("Original Image", image)
cv2.imshow("Canny", canny)
cv2.imshow("laplacian", canny)
cv2.imshow("Sobel", sobel)
cv2.imshow("Prewitt", prewittx + prewitty)

