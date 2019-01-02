#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from yoloOpencv import opencvYOLO
import cv2
import imutils
import time
from libPOS import desktop

yolo = opencvYOLO(modeltype="yolov3-tiny", \
    objnames="cfg.breads_v3.tiny/obj.names", \
    weights="cfg.breads_v3.tiny/weights/yolov3-tiny_500000.weights",\
    cfg="cfg.breads_v3.tiny/yolov3-tiny.cfg")

labels = {  "a1":["杯子蛋糕", 15], "a2":["丹麥起司",55], "a3":["十勝紅豆",42], "a4":["花生夾心", 26], \
            "a5":["夾心鬆餅", 42], "a6":["風味布雪", 38] }
idle_checkout = (6, 10)
media = "bread_test.mp4"
video_out = "output.avi"
dt = desktop("images/bg.jpg", "images/bgClick.jpg")

cv2.namedWindow("SunplusIT", cv2.WND_PROP_FULLSCREEN)        # Create a named window
cv2.setWindowProperty("SunplusIT", cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

start_time = time.time()
dt.emptyBG = None
last_movetime = time.time()  #objects > 0
YOLO = False  # YOLO detect in this loop?
txtStatus = ""

def group(items):
    """
    groups a sorted list of integers into sublists based on the integer key
    """
    if len(items) == 0:
        return []

    items.sort()
    grouped_items = []
    prev_item, rest_items = items[0], items[1:]

    subgroup = [prev_item]
    for item in rest_items:
        if item != prev_item:
            grouped_items.append(subgroup)
            subgroup = []
        subgroup.append(item)
        prev_item = item

    grouped_items.append(subgroup)
    return grouped_items

if __name__ == "__main__":

    INPUT = cv2.VideoCapture(0)

    width = int(INPUT.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(INPUT.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_out,fourcc, 30.0, (int(width),int(height)))

    frameID = 0
    while True:
        hasFrame, frame = INPUT.read()
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("--- %s seconds ---" % (time.time() - start_time))
            break


        '''
        yolo.getObject(frame, labelWant="", drawBox=True, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255))
        print ("Object counts:", yolo.objCounts)
        #yolo.listLabels()
        print("classIds:{}, confidences:{}, labelName:{}, bbox:{}".\
            format(len(yolo.classIds), len(yolo.scores), len(yolo.labelNames), len(yolo.bbox)) )
        #cv2.imshow("Frame", imutils.resize(frame, width=600))
        '''

        #objects = dt.getContours(frame, 1200)
        #print("Objects:", objects)

        if(dt.emptyBG is None or time.time()-dt.emptyBG_time>=0.5):
            dt.emptyBG = frame.copy()
            dt.emptyBG_time = time.time()
            #print("Update BG")

        objects = dt.difference(dt.emptyBG, frame, 1000)
        if(objects>0):
            last_movetime = time.time()
            timeout_move = str(round(time.time()-last_movetime, 0))
            txtStatus = "move:" + timeout_move
        else:
            waiting = time.time() - last_movetime
            timeout_move = str(round(time.time()-last_movetime, 0))
            txtStatus = "move:" + timeout_move

            if(waiting > idle_checkout[0] and waiting<idle_checkout[1] ):
                txtStatus = "detecting"
                YOLO = True

        imgDisplay = dt.display(frame.copy(), txtStatus)
        cv2.imshow("SunplusIT", imgDisplay)
        cv2.waitKey(1)

        if(YOLO is True):
            print("YOLO start...")
            YOLO = False
            yolo.getObject(frame, labelWant="", drawBox=True, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255))
            #yolo.listLabels()
            #print("classIds:{}, confidences:{}, labelName:{}, bbox:{}".\
            #    format(len(yolo.classIds), len(yolo.scores), len(yolo.labelNames), len(yolo.bbox)) )
            if(len(yolo.labelNames)>0):
                types = group(yolo.labelNames)
                print("Labels:", types)
                shoplist = []
                for items in types:
                    shoplist.append([items[0], labels[items[0]][0], labels[items[0]][1], len(items)])

                txtStatus = "checkout"
                print(shoplist)

                imgDisplay = dt.display(frame, txtStatus, shoplist)
                cv2.imshow("SunplusIT", imgDisplay)
                cv2.waitKey(0)
                #cv2.imshow("SunplusIT", imgDisplay)
                #cv2.waitKey(1)
                #time.sleep(0)

        #dt.emptyBG = frame.copy()
        #dt.emptyBG_time = time.time()
        #if(video_out!=""):
        #out.write(frame)
        

        k = cv2.waitKey(0)
        if k == 0xFF & ord("q"):
            out.release()
            break
