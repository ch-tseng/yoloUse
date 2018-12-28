import time
import imutils
import cv2
import numpy as np

class desktop:
    def __init__(self, bg_path, click_bg_path, windowName="frame"):
        self.bg = cv2.imread(bg_path)
        self.win = windowName
        self.click_bg = cv2.imread(click_bg_path)

    def display(self, camImg, itemList=None):
        resized = cv2.resize(camImg, (480, 360))
        bg = self.bg
        print(bg.shape, resized.shape)
        bg[95:95+resized.shape[0],25:25+resized.shape[1]] = resized
        cv2.imshow(self.win, bg)
        cv2.waitKey(1)
