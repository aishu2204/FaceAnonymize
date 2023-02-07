from utils.face_blurring import anonymize_face_simple
from utils.face_blurring import anonymize_face_pixelate
import cv2
import numpy as np
import os
import argparse

# constructing argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-f', '--face', required=True, help='path to face detector model directory')
ap.add_argument('-m', '--method', type=str, default='simple', \
                choices=['simple','pixelated'], help='face blurring/anonymizing method')
ap.add_argument('-b', '--blocks', type=int, default=20, help='number of blocks \
                                                             for pixelated blurring method')
ap.add_argument('-c', '--confidence', type=float, default=0.5, \
                help='minimum probability to filter weak detections')

args = vars(ap.parse_args())

# loading face detector model

print('[INFO] Loading face detector model...')
prototxtPath = os.path.sep.join([args['face'], 'deploy.prototxt'])
weightsPath = os.path.sep.join([args['face'], 'res10_300x300_ssd_iter_140000.caffemodel'])

net = cv2.dnn.readNet(prototxtPath, weightsPath)

# loading input image from disk, cloning it and grabbing img dims
img = cv2.imread(args['image'])
orig = img.copy()
(h, w) = img.shape[:2]

# constructing blob from image
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0)) # 1.0 is scaling factor, (300, 300) is size of the input image that nn expects, mean RGB values

# pass blob through the network to obtain face detections
print('[INFO] computing face detections...')
net.setInput(blob)
detections = net.forward()

# loop over the detections
# https://towardsdatascience.com/step-by-step-face-recognition-code-implementation-from-scratch-in-python-cc95fa041120

for i in range(0, detections.shape[2]):
    # extract confidence associated with detection
    confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring that confidence is greater than
    # min confidence
    if confidence > args['confidence']:
        # compute bounding box coordinates of face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')

        # extracting face roi
        face = img[startY:endY, startX:endX]

        # check argument method
        if args['method'] == 'simple':
            face = anonymize_face_simple(face, factor=3.0)
        else:
            face = anonymize_face_pixelate(face, blocks=args['blocks'])

        # stored blurred face in output image
        img[startY:endY, startX:endX] = face

# display original and blurred image side by side
output = np.hstack([orig, img])
cv2.imshow('Blurred Vs Original', output)
cv2.waitKey(0)


