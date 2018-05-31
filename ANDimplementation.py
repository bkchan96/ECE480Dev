import numpy as np
import cv2

#set parameters for the simple blob detector to detect shapes similar to scantron filled-in answer choices
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

#Reading in the submission and answer key and resizing to standardize images for masking
scan_submission = cv2.imread('C:/Users/jaredh/Desktop/College/ECE480/ECE480 Scantron Reader/submission1.jpg')
scan_submission = cv2.resize(scan_submission, (800,1400))
scan_answerkey = cv2.imread('C:/Users/jaredh/Desktop/College/ECE480/ECE480 Scantron Reader/answerkey.jpg')
scan_answerkey = cv2.resize(scan_answerkey, (800,1400))

#Converts the submission image into a binary image with answer choices circled
#Function Explanations:
#cvtColor() - Converts an RGB image to grayscale
#Gaussianblur() - Blends in all pixels in the image by a set parameter to eliminate noise
#adaptiveThreshold() - Works similarly to thresholding in that the pixels on the image are sorted between 
    #two levels of intensity: white and black. However, adaptive thresholding does this on a pixel-by-pixel
    #basis instead of image-by-image. Thus, the edges are detected and clearly defined
#SimpleBlobDetector() - instantiates a simple blob detector that specifically looks for blobs on the image
    #that match the parameters being passed into it
#detect() - Scans the image and saves the blobs that match with the parameters in a data structure
    #called a 'keypoint'. These keypoints are then saved in an array for later use
#drawKeypoints() - Takes the keypoints found in the detect() function and draws them on the 
    #image fed into the function. 
#len() - Finds the length of the keypoints array, which represents the number of blobs detected (i.e., filled-in answers)
scan_sub_mask = cv2.cvtColor(scan_submission,cv2.COLOR_BGR2GRAY)
scan_sub_mask = cv2.GaussianBlur(scan_sub_mask, (5,5), 0)
scan_sub_mask = cv2.adaptiveThreshold(scan_sub_mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 21, 10)
subdetector = cv2.SimpleBlobDetector(params)
subkeypoints = subdetector.detect(scan_sub_mask)
sub_keypoints_drawn = cv2.drawKeypoints(scan_sub_mask, subkeypoints, 
        np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Submission',sub_keypoints_drawn)
submission = len(subkeypoints)
print submission

#Performs the same functions as in the submission image, but for the answer key instead
scan_answer_mask = cv2.cvtColor(scan_answerkey,cv2.COLOR_BGR2GRAY)
scan_answer_mask = cv2.GaussianBlur(scan_answer_mask, (5,5), 0)
scan_answer_mask = cv2.adaptiveThreshold(scan_answer_mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 21, 10)
ansdetector = cv2.SimpleBlobDetector(params)
anskeypoints = ansdetector.detect(scan_answer_mask)
ans_keypoints_drawn = cv2.drawKeypoints(scan_answer_mask, anskeypoints, 
        np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Answer Key',ans_keypoints_drawn)
corrans= len(anskeypoints)
print corrans

#Concatenates the two processed images together by performing a bitwise AND operation
#Wrong answers are detected as extra keypoints and correct answers are combined with the answer key
    #and read as one answer (essentially ignored)
#Keypoints are drawn on this image and compared with the keypoints on the submission. Assuming an
    #ideal answer key scantron, the number of keypoints left after the submission keypoints are subtracted 
    #from the total keypoints to output the number of incorrect answers
AND_MASK = (scan_sub_mask & scan_answer_mask)
cropped = cv2.resize(AND_MASK, (800, 1400))
totaldetector = cv2.SimpleBlobDetector(params)
totalkeypoints = totaldetector.detect(cropped)
finalkeypoints_drawn = cv2.drawKeypoints(cropped, totalkeypoints, np.array([]),
        (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
nblobs = len(totalkeypoints)
print nblobs
print str(nblobs - submission) + " incorrect answers"
incorr = str(nblobs - submission)

#Establishes a GUI to display the number of incorrect answers, and can be personalized by the user
back = cv2.imread('C:/Users/jaredh/Desktop/College/ECE480/ECE480 Scantron Reader/back.png',1)
cv2.putText(back, "You have " + incorr + " incorrect answers",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
cv2.imshow('AND masking', finalkeypoints_drawn)
cv2.imshow('GUI',back)
cv2.waitKey(0)