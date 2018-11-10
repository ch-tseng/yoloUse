from yoloOpencv import opencvYOLO
import cv2
import imutils
import time

yolo = opencvYOLO(modeltype="yolov3", \
    objnames="/media/sf_ShareFolder/cfg.roadYolo/obj.names", \
    weights="/media/sf_ShareFolder/cfg.roadYolo/yolov3_30000.weights",\
    cfg="/media/sf_ShareFolder/cfg.roadYolo/yolov3.cfg")

inputType = "video"  # webcam, image, video
media = "/media/sf_ShareFolder/road.mp4"
video_out = "/media/sf_ShareFolder/road.avi"

start_time = time.time()

if __name__ == "__main__":

    if(inputType == "webcam"):
        INPUT = cv2.VideoCapture(0)
    elif(inputType == "image"):
        INPUT = cv2.imread(media)
    elif(inputType == "video"):
        INPUT = cv2.VideoCapture(media)

    if(inputType == "image"):
        yolo.getObject(INPUT, labelWant="", drawBox=True, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255))
        print ("Object counts:", yolo.objCounts)
        print("classIds:{}, confidences:{}, labelName:{}, bbox:{}".\
        format(len(yolo.classIds), len(yolo.scores), len(yolo.labelNames), len(yolo.bbox)) )
        cv2.imshow("Frame", imutils.resize(INPUT, width=850))

        k = cv2.waitKey(0)
        if k == 0xFF & ord("q"):
            out.release()

    else:
        if(video_out!=""):
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

            yolo.getObject(frame, labelWant="", drawBox=True, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255))
            print ("Object counts:", yolo.objCounts)
            #yolo.listLabels()
            print("classIds:{}, confidences:{}, labelName:{}, bbox:{}".\
                format(len(yolo.classIds), len(yolo.scores), len(yolo.labelNames), len(yolo.bbox)) )
            cv2.imshow("Frame", imutils.resize(frame, width=850))

            if(video_out!=""):
                out.write(frame)

            k = cv2.waitKey(1)
            if k == 0xFF & ord("q"):
                out.release()
                break
