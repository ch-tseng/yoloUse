import time
import imutils
import cv2
import numpy as np
from skimage.measure import compare_ssim

class desktop:
    def __init__(self, bg_path, click_bg_path):
        self.bg = cv2.imread(bg_path)
        self.click_bg = cv2.imread(click_bg_path)

    def display(self, camImg, txtStatus=None, itemList=None):
        resized = cv2.resize(camImg, (480, 360))
        bg = self.bg
        print(bg.shape, resized.shape)
        bg[95:95+resized.shape[0],25:25+resized.shape[1]] = resized

        if(txtStatus is not None):
            cv2.putText(bg, txtStatus, (290,120), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,255,0), 2)

        return bg
        #cv2.imshow(self.win, bg)
        #cv2.waitKey(1)

    def getContours(self, img, minSize=1600):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        gray = cv2.Canny(gray, 75, 200)
        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        #cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:7]
        cv2.imshow("TEST", gray)
        #keeps = []
        # loop over the contours again
        counts = 0
        for (i, c) in enumerate(cnts):
            # compute the area and the perimeter of the contour
            area = cv2.contourArea(c)
            #print(area)
            if(area>minSize):
                counts += 1
                #print(area)
                #keeps.append(c)

        #cv2.drawContours(img, keeps, -1, (0, 255, 0), 2)
        print("Found {} contours".format(len(cnts)))

        #cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
        return counts

    def difference(self, img1, img2, minSize=1200):
        sensitive_th = 90

        img1 = img1[50:img1.shape[0]-50, 50:img1.shape[1]-50]
        img2 = img2[50:img2.shape[0]-50, 50:img2.shape[1]-50]
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (15, 15), 0)
        gray2 = cv2.GaussianBlur(gray2, (15, 15), 0)

        diff = cv2.subtract(gray1, gray2)
        #cv2.imshow("TEST", diff)
        thresh = cv2.threshold(diff, sensitive_th, 255, cv2.THRESH_BINARY_INV)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        #(score, diff) = compare_ssim(gray1, gray2, full=True)
        #diff = (diff * 255).astype("uint8")
        #print("SSIM: {}".format(score))

        #thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        #cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = imutils.grab_contours(cnts)

        # loop over the contours
        counts = 0
        for c in cnts:
            # compute the bounding box of the contour and then draw the
            # bounding box on both input images to represent where the two
            # images differ
            area = cv2.contourArea(c)
            print(area)
            if(area>minSize and area<(img1.shape[0]-20)*(img1.shape[1]-20)):
                counts += 1
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
        # show the output images
        #cv2.imshow("Original", img1)
        #cv2.imshow("Modified", img2)
        #cv2.imshow("Diff", diff)
        #cv2.imshow("Thresh", thresh)

        return counts
