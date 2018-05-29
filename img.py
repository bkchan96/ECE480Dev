import numpy as np
import cv2
import matplotlib
import imutils
import matplotlib.pyplot as plt

# import matplotlib.patches as patches

# img = cv2.imread('pKjxCMk.jpg',0)
# print img
# def drawMatches(img1, kp1, img2, kp2, matches):
#        rows1 = img1.shape[0]
#        cols1 = img1.shape[1]
#        rows2 = img2.shape[0]
#        cols2 = img2.shape[1]
#
#        out = np.zeros((max([rows1,rows2]),cols1+cols2,3),dtype = 'uint8')
#        out[:rows1, :cols1] = np.dstack([img1])
#        out[:rows2, :cols2] = np.dstack([img2])
#
#        for mat in matches:
#            img1_idx = mat.queryIdx
#            img2_idx = mat.trainIdx
#            (x1,y1) = kp1[img1_idx].pt
#            (x2,y2) = kp2[img2_idx].pt
#            cv2.circle(out,(int(x1),int(y1)),4,(255,0,0),1)
#            cv2.circle(out, (int(x2)+cols1,int(y2)),4,(255,0,0),1)
#           cv2.line(out,(int(x1),int(y1)),(int(x2)+cols1,int(y2)),(255,0,0),1)
#       return out

camera = cv2.VideoCapture(0)
# fgbg = cv2.BackgroundSubtractorMOG()
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

answer = cv2.imread('answerkey.jpg', 0)
akey = cv2.GaussianBlur(answer, (5, 5), 0)
akey = cv2.adaptiveThreshold(akey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
akey = cv2.Canny(akey, 20, 100)
detector = cv2.SimpleBlobDetector_create(params)
akeypoints = detector.detect(akey)
answerkeypoints = cv2.drawKeypoints(akey, akeypoints, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('answer key', answerkeypoints)

while True:
    return_value1, image = camera.read()
    RGB = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # RGB = fgbg.apply(image)
    RGB = cv2.GaussianBlur(RGB, (5, 5), 0)
    RGB = cv2.adaptiveThreshold(RGB, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
    RGB = cv2.Canny(RGB, 20, 100)
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(RGB)
    im_with_keypoints = cv2.drawKeypoints(RGB, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # display = cv2.drawMatches(RGB, keypoints, akey, akeypoints, matches[:10], np.array([]), (0, 255, 0), (0, 0, 255), None, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # display = drawMatches(RGB, keypoints, akey, akeypoints, np.array[])
    # RGB = cv2.bitwise_not(RGB)
    # cv2.rectangle(RGB,(50,100),(80,116),(0,0,255),1)
    # cv2.rectangle(RGB,(50,415),(80,425),(0,0,255),1)        #            cv2.imwrite('answerkey.jpg', image)

    # # Read source image.
    # im_src = cv2.imread('book2.jpg')
    # # Four corners of the book in source image
    # pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])
    #
    # # Read destination image.
    # im_dst = cv2.imread('book1.jpg')
    # # Four corners of the book in destination image.
    # pts_dst = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])
    #
    # # Calculate Homography
    # h, status = cv2.findHomography(pts_src, pts_dst)
    #
    # # Warp source image to destination based on homography
    # im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

    #paper = four_point_transform(image, docCnt.reshape(4, 2))


    cv2.imshow('image', im_with_keypoints)
    cv2.imshow('answer key', answerkeypoints)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('camcam.jpg', im_with_keypoints)

        break
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
camera.release()
cv2.destroyAllWindows()



