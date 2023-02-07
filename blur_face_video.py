from utils.face_blurring import anonymize_face_simple
from utils.face_blurring import anonymize_face_pixelate
import numpy as np
import cv2
import argparse
import os
from imutils.video import VideoStream
import time
import imutils

# constructing argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', required=True, help='Path to face detector model dir')
ap.add_argument('-m', '--method', type=str, default='simple',\
                choices=['simple','pixelated'], help='Type of anonymization')
ap.add_argument('-b', '--blocks', type=int, default=20, \
                help='No of blocks for pixelated anonymization')
ap.add_argument('-c', '--confidence', type=float, default=0.5, \
                help='min probability to filter detections')

args = vars(ap.parse_args())

# loading face detector model

print('[INFO] Loading face detector model...')
prototxtPath = os.path.sep.join([args['face'], 'deploy.prototxt'])
weightsPath = os.path.sep.join([args['face'], 'res10_300x300_ssd_iter_140000.caffemodel'])

net = cv2.dnn.readNet(prototxtPath, weightsPath)


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            # check to see if we are applying the "simple" face
            # blurring method
            if args["method"] == "simple":
                face = anonymize_face_simple(face, factor=3.0)
            # otherwise, we must be applying the "pixelated" face
            # anonymization method
            else:
                face = anonymize_face_pixelate(face,
                                               blocks=args["blocks"])
            # store the blurred face in the output image
            frame[startY:endY, startX:endX] = face



    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    # if the `q` key was pressed, break from the loop
    if key == 27:
        break


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()