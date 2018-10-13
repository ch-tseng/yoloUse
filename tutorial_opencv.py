from yoloOpencv import opencvYOLO
import cv2
import imutils
import time

yolo = opencvYOLO(modeltype="yolov3", objnames="../darknet/data/coco.names",
    weights="../darknet/weights/yolov3.weights", cfg="../darknet/cfg/yolov3.cfg")

start_time = time.time()

if __name__ == "__main__":

    #VIDEO_IN = cv2.VideoCapture(0)
    VIDEO_IN = cv2.VideoCapture("/media/sf_ShareFolder/traffic.mp4")

    frameID = 0
    while True:
        hasFrame, frame = VIDEO_IN.read()
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("--- %s seconds ---" % (time.time() - start_time))
            break

        yolo.getObject(frame, labelWant="", drawBox=True, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255))
        print ("Object counts:", yolo.objCounts)
        #yolo.listLabels()
        print("classIds:{}, confidences:{}, labelName:{}, bbox:{}".\
            format(len(yolo.classIds), len(yolo.scores), len(yolo.labelNames), len(yolo.bbox)) )
        cv2.imshow("Frame", imutils.resize(frame, width=850))

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            out.release()
            break
