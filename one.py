# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
#from skimage.measure import structural_similarity as ssim

# im = cv2.imread('answerkey.jpg')
#
# #r = cv2.selectROI(im)
#
# # Crop image
# #imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
#
# # Display cropped image
# cv2.rectangle(im, (420, 205), (595, 385),
#     (0, 0, 255), 1)
# cv2.imshow("Image", im)
#
#
# # Wait for Esc key to stop
# k = cv2.waitKey(5) & 0xFF
#
# if k%256 == 32:
# # cpture images when SPACE pressed
# #save image as
# #org_name = "C:/Users/phant/Desktop/opencv_frame.png".format(img_counter)
# cv2.imwrite("pic.png",im)
#

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 10
params.maxThreshold = 200
params.filterByArea = True
params.minArea = 100
params.maxArea = 200
params.filterByCircularity = True
params.minCircularity = 0.1
params.filterByConvexity = True
params.minConvexity = 0.87
params.filterByInertia = True
params.minInertiaRatio = 0.01



imagea = cv2.imread('answerkey.jpg')
imageb = cv2.imread('answerkeyfilled.jpg')
RGB1 = cv2.cvtColor(imagea,cv2.COLOR_BGR2GRAY)
   # RGB = fgbg.apply(image)
RGB1 = cv2.GaussianBlur(RGB1, (5,5), 0)
RGB1 = cv2.adaptiveThreshold(RGB1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

RGB = cv2.cvtColor(imageb,cv2.COLOR_BGR2GRAY)
   # RGB = fgbg.apply(image)
RGB = cv2.GaussianBlur(RGB, (5,5), 0)
RGB = cv2.adaptiveThreshold(RGB, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

# detector1 = cv2.SimpleBlobDetector_create(params)
# akeypoints1 = detector1.detect(RGB1)
# answerkeypoints1 = cv2.drawKeypoints(RGB1, akeypoints1, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# #mask = np.zeros
# imageA = cv2.cvtColor(imagea, cv2.COLOR_BGR2GRAY)
# imageB = cv2.cvtColor(imageb, cv2.COLOR_BGR2GRAY)


s = cv2.bitwise_xor(RGB1, RGB)
cropped = cv2.resize(s, (1600, 900))
cropped = cv2.bitwise_not(cropped)
#cv2.imshow('res', cropped)
detector = cv2.SimpleBlobDetector_create(params)
akeypoints = detector.detect(cropped)
answerkeypoints = cv2.drawKeypoints(cropped, akeypoints, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow('res', cv2.resize((imageA & imageB),(1600,900)))
cv2.imshow('res', answerkeypoints)

cv2.waitKey(0)