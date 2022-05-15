import numpy as np
from matplotlib import pyplot as plt
import cv2 #Opencv
import math
import skimage
from skimage import io

img = cv2.imread('monedas.jpg')
rimg = cv2.resize(img, (300,300))
cv2.imshow('Imagen Original',rimg)

#Canny

img_g = cv2.cvtColor(rimg,cv2.COLOR_BGR2GRAY)
img_s = cv2.GaussianBlur(img_g,(5,5),0)
img_b = cv2.Canny(img_s, 150, 200)
cv2.imshow('Canny',img_b)

#Laplace

lap = cv2.Laplacian(img_g, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
cv2.imshow("Laplaciano",lap)

#Sobel

sobelX = cv2.Sobel(img_g,cv2.CV_64F,1,0)
sobelY = cv2.Sobel(img_g,cv2.CV_64F,0,1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

#SobelX

cv2.imshow("SobelX",sobelX)

#SobelY

cv2.imshow("SobelY",sobelY)

#Sobel Combined

sobelCombined = cv2.bitwise_or(sobelX, sobelY)
cv2.imshow("Sobel",sobelCombined)

cv2.waitKey(0)
cv2.destroyAllWindows()
