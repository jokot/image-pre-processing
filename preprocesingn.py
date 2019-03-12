import cv2
import numpy as np

def init_data(path):
    image = cv2.imread(path)
    cv2.imshow("original",image)    
    return image

def preprocessing(image):
##scalling
    image = cv2.resize(image,None,fx=0.6,fy=0.6)
    cv2.imshow("greyscale",image)
    image1 = image
    image2 = image
##grey scale
##    greyScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
##    cv2.imshow("greyscale",greyScale)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    cv2.imshow("greyscale",image)

##image negative
    image = cv2.bitwise_not(image)
    cv2.imshow("negative",image)

##contrast streching
    minv = np.amin(image)
    maxv = np.amax(image)
    h, w = image.shape[:2]
##    contrastStrach = contrast_straching(image,maxv,minv,h,w)
##    cv2.imshow("contrast Streching",contrastStrach)
    image1 = contrast_straching(image1,maxv,minv,h,w)
    cv2.imshow("contrast Streching",image1)

    
##histogram equ
##    image = histogram_equalize(image)
##    image = histogram_equalize(contrastStrach)
##    cv2.imshow("histogram equalization",image)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    equ = histogram_equalize(image2)
    cv2.imshow("histogram equalization",equ)
    
##binary image
##  binary = binary_image(equ,h,w)
##  cv2.imshow("binnary segmentation",binary)
    image = binary_image(equ,h,w)
    cv2.imshow("binnary segmentation",image)

    


def contrast_straching(image,maxv,minv,h,w):    
    result = image
    for i in range(0,h-1,1):
        for j in range(0,w-1,1):
            temp = ((image[i,j]-minv)/(maxv-minv))*255
            result[i,j]= temp
    return result


def histogram_equalize(image):
    equ = cv2.equalizeHist(image)
    return equ


def binary_image(image,h,w):
    binImage = image
    T=100
    for i in range(h-1):
        for j in range(w-1):
            if(binImage[i,j]<100):
                binImage[i,j] = 0
            else:
                binImage[i,j] = 255    
    return binImage

##main Program ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
image = init_data('traffic.png')
preprocessing(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
