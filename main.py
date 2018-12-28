from yoloOpencv import opencvYOLO
import cv2
import imutils
import time
from libPOS import desktop

yolo = opencvYOLO(modeltype="yolov3-tiny", \
    objnames="cfg.breads_v3.tiny/obj.names", \
    weights="cfg.breads_v3.tiny/weights/yolov3-tiny_200000.weights",\
    cfg="cfg.breads_v3.tiny/yolov3-tiny.cfg")


media = "bread_test.mp4"
video_out = "output.avi"
dt = desktop("images/bg.jpg", "images/bgClick.jpg", "SunplusIT")

cv2.namedWindow("SunplusIT", cv2.WND_PROP_FULLSCREEN)        # Create a named window
cv2.setWindowProperty("SunplusIT", cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

start_time = time.time()


if __name__ == "__main__":

    INPUT = cv2.VideoCapture(media)

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
        #cv2.imshow("Frame", imutils.resize(frame, width=600))
        dt.display(frame)
        #if(video_out!=""):
        #out.write(frame)

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            out.release()
            break
