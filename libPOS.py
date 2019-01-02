import time
from PIL import ImageFont, ImageDraw, Image
import imutils
import cv2
import numpy as np
from skimage.measure import compare_ssim

class desktop:
    def __init__(self, bg_path, click_bg_path):
        self.bg = bg_path
        self.click_bg = cv2.imread(click_bg_path)

    def display(self, camImg, txtStatus=None, itemList=None):
        itemList_pos = (30, 550)
        itemList_h = 40
        bg = cv2.imread(self.bg)

        resized = cv2.resize(camImg, (480, 360))
        print(bg.shape, resized.shape)
        bg[95:95+resized.shape[0],25:25+resized.shape[1]] = resized

        if(txtStatus is not None):
            cv2.putText(bg, txtStatus, (290,120), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,255,0), 2)

        price_total = 0
        if(itemList is not None):
            y = itemList_pos[0]
            for id, item in enumerate(itemList):
                '''
                txtIMG = cv2.imread("images/products/" + item[0] + ".jpg")

                bg[y:y+txtIMG.shape[0], itemList_pos[1]:itemList_pos[1]+txtIMG.shape[1]] = txtIMG
                y += itemList_h
                '''

                bg = self.printText(item[1], bg=bg, color=(0,0,0,0), size=0.7, pos=(itemList_pos[1],y), type="Chinese")

                cv2.putText(bg, str(id+1)+")", (itemList_pos[1]-30,y+17), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)
                if(item[3]>1):
                    cv2.putText(bg, "x "+str(item[3]), (itemList_pos[1]+130,y+17), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)

                price = item[2]*item[3]
                cv2.putText(bg, "$"+str(price), (itemList_pos[1]+180,y+17), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,0,0), 1)

                price_total += price

            total = cv2.imread("images/total.jpg")
            bg[410:410+total.shape[0], 515:515+total.shape[1]] = total
            cv2.putText(bg, "$"+str(price_total), (660,395+total.shape[0]), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,0,0), 2)
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

    def printText(self, txt, bg, color=(0,255,0,0), size=0.7, pos=(0,0), type="Chinese"):
        (b,g,r,a) = color

        if(type=="English"):
            ## Use cv2.FONT_HERSHEY_XXX to write English.
            cv2.putText(bg,  txt, pos, cv2.FONT_HERSHEY_SIMPLEX, size,  (b,g,r), 2, cv2.LINE_AA)

        else:
            ## Use simsum.ttf to write Chinese.
            fontpath = "fonts/wt009.ttf"
            print("TEST", txt)
            font = ImageFont.truetype(fontpath, int(size*10*4))
            img_pil = Image.fromarray(bg)
            draw = ImageDraw.Draw(img_pil)
            draw.text(pos,  txt, font = font, fill = (b, g, r, a))
            bg = np.array(img_pil)

        return bg
