from yoloOpencv import opencvYOLO
import cv2
import imutils

yolo = opencvYOLO(modeltype="yolov3", objnames="coco.names", weights="yolov3.weights",
yolo = opencvYOLO(modeltype="yolov3", objnames=""coco.names"",
    weights="yolov3.weights", cfg="../darknet/cfg/yolov3.cfg")

if __name__ == "__main__":

    VIDEO_IN = cv2.VideoCapture("/media/sf_ShareFolder/IMG_8055.m4v")

    frameID = 0
    while True:
        hasFrame, frame = VIDEO_IN.read()
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            cv2.waitKey(3000)
            break

        yolo.getObject(frame, labelWant="", drawBox=True)
        print ("Object counts:", yolo.objCounts)
        yolo.listLabels()
        cv2.imshow("Frame", imutils.resize(frame, width=850))
        cv2.waitKey(1)

