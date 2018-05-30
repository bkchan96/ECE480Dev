import numpy as np
import cv2
import matplotlib
import imutils
import matplotlib.pyplot as plt

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 80
params.maxThreshold = 100
params.filterByArea = True
params.minArea = 600
params.maxArea = 1000
params.filterByCircularity = True
params.minCircularity = 0.4
params.maxCircularity = 0.8
params.filterByConvexity = True
params.minConvexity = 0.5
params.filterByInertia = True
params.minInertiaRatio = 0.01

imagea = cv2.imread('answerkeyfilled2.jpg')
imagea = cv2.resize(imagea, (800,1400))
imageb = cv2.imread('answerkeyfilled3.jpg')
imageb = cv2.resize(imageb, (800,1400))

RGB1 = cv2.cvtColor(imagea,cv2.COLOR_BGR2GRAY)
   # RGB = fgbg.apply(image)
RGB1 = cv2.GaussianBlur(RGB1, (5,5), 0)
RGB1 = cv2.adaptiveThreshold(RGB1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
#RGB1 = cv2.GaussianBlur(RGB1, (5,5), 1)
detector2 = cv2.SimpleBlobDetector(params)
akeypoints2 = detector2.detect(RGB1)
answerkeypoints2 = cv2.drawKeypoints(RGB1, akeypoints2, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('submission',answerkeypoints2)
submission = len(akeypoints2)

RGB = cv2.cvtColor(imageb,cv2.COLOR_BGR2GRAY)
   # RGB = fgbg.apply(image)
RGB = cv2.GaussianBlur(RGB, (5,5), 0)
RGB = cv2.adaptiveThreshold(RGB, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
#RGB = cv2.GaussianBlur(RGB, (5,5), 1)
detector3 = cv2.SimpleBlobDetector(params)
akeypoints3 = detector3.detect(RGB)
answerkeypoints3 = cv2.drawKeypoints(RGB, akeypoints3, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('answersheet',answerkeypoints3)
corrans= len(akeypoints3)

# detector1 = cv2.SimpleBlobDetector_create(params)
# akeypoints1 = detector1.detect(RGB1)
# answerkeypoints1 = cv2.drawKeypoints(RGB1, akeypoints1, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# #mask = np.zeros
# imageA = cv2.cvtColor(imagea, cv2.COLOR_BGR2GRAY)
# imageB = cv2.cvtColor(imageb, cv2.COLOR_BGR2GRAY)


s = (RGB1 & RGB)
cropped = cv2.resize(s, (800,1400))
##cropped = cv2.bitwise_not(cropped)
#cropped = cv2.GaussianBlur(cropped, (5,5), 1)
detector = cv2.SimpleBlobDetector(params)
akeypoints = detector.detect(cropped)
answerkeypoints = cv2.drawKeypoints(cropped, akeypoints, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
nblobs = len(akeypoints)
print str(nblobs - submission) + " incorrect answers"
#incorr = (nblobs - submission)
#gui menu 
#back = cv2.imread('C:\Users\jaredh\Desktop\back.png',1)
#cv2.putText(back, incorr + "incorrect answers",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,50), 2)
#cv2.imshow('Menu', back)


#cv2.imshow('res', cv2.resize((imageA & imageB),(1600,900)))
cv2.imshow('res', answerkeypoints)
cv2.waitKey(0)