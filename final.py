import numpy as np
import imutils
import cv2

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

######################################################
# blob detection parameters
######################################################
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 10 #10
params.maxThreshold = 200 #200
params.filterByArea = True #True
params.minArea = 400
params.maxArea = 1000
params.filterByCircularity = False
params.minCircularity = 0.0
params.filterByConvexity = False #True
params.minConvexity = 0.87
params.filterByInertia = False
params.minInertiaRatio = 0.01

############################################################################
# load the image, convert it to grayscale, blur it slightly, then find edges
############################################################################
image1 = cv2.imread('answerkeyfilled.jpg')
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
blurred1 = cv2.GaussianBlur(gray1, (5, 5), 0)
black1 = cv2.adaptiveThreshold(blurred1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
edged1 = cv2.Canny(black1,75, 200)
cropped11 = cv2.resize(edged1, (1600, 900))

############################################################################
# find contours#############################################################
############################################################################
cnts1 = cv2.findContours(edged1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts1 = cnts1[0] if imutils.is_cv2() else cnts1[1]
docCnt1 = None
# ensure that at least one contour was found
if len(cnts1) > 0:
    # sort the contours according to their size in
    # descending order
    cnts1 = sorted(cnts1, key=cv2.contourArea, reverse=True)

    # loop over the sorted contours
    for c in cnts1:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points,
        # then we can assume we have found the paper
        if len(approx) == 4:
            docCnt1 = approx
            break

############################################################################
# load the image, convert it to grayscale, blur it slightly, then find edges
############################################################################
image2 = cv2.imread('answerkey.jpg')
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
blurred2 = cv2.GaussianBlur(gray2, (5, 5), 0)
black2 = cv2.adaptiveThreshold(blurred2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
edged2 = cv2.Canny(black2,75, 200)
cropped22 = cv2.resize(edged2, (1600, 900))

############################################################################
# find contours#############################################################
############################################################################
cnts2 = cv2.findContours(edged2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts2 = cnts2[0] if imutils.is_cv2() else cnts2[1]
docCnt2 = None
# ensure that at least one contour was found
if len(cnts2) > 0:
    # sort the contours according to their size in
    # descending order
    cnts2 = sorted(cnts2, key=cv2.contourArea, reverse=True)

    # loop over the sorted contours
    for c in cnts2:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points,
        # then we can assume we have found the paper
        if len(approx) == 4:
            docCnt2 = approx
            break


warped1 = four_point_transform(gray1, docCnt1.reshape(4, 2))
warped2 = four_point_transform(gray2, docCnt2.reshape(4, 2))

######################################################
# prepare both images
######################################################
RGB1 = cv2.GaussianBlur(warped1, (5,5), 0)
RGB1 = cv2.adaptiveThreshold(RGB1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
cropped1 = cv2.resize(RGB1, (500, 1000))

RGB2 = cv2.GaussianBlur(warped2, (5,5), 0)
RGB2 = cv2.adaptiveThreshold(RGB2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
cropped2 = cv2.resize(RGB2, (500, 1000))

######################################################
# xor images and prepare for blob detection
######################################################
s = ~(RGB1 ^ RGB2)
s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
s = cv2.GaussianBlur(s, (75,75), 10)


######################################################
# run blob detection
######################################################
detector = cv2.SimpleBlobDetector_create(params)
akeypoints = detector.detect(s)
answerkeypoints = cv2.drawKeypoints(s, akeypoints, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('result', answerkeypoints)

numberofincorrect = int(len(akeypoints)/2)
print(numberofincorrect)

#gui menu
back = cv2.imread('image.jpg',1)
back = cv2.resize(back, (450, 50))
cv2.putText(back, "This Student has " + str(numberofincorrect) + " Errors",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,50,50), 2)
cv2.imshow('Menu', back)

cv2.waitKey(0)
