# import the necessary packages
import cv2
import numpy as np


def anonymize_face_simple(image, factor=3.0):
    # automatically determine the size of the blurring kernel based
    # on the spatial dimensions of the input image
    (h, w) = image.shape[:2]
    kW = int(h / factor)
    kH = int(w / factor)

    # ensure the width of the kernel is odd
    if kW % 2 == 0:
        kW -= 1

    # ensure the height of the kernel is odd
    if kH % 2 == 0:
        kH -= 1

    # apply gaussian blur with the computed kernel size
    return cv2.GaussianBlur(image, (kW, kH), 0)


def anonymize_face_pixelate(image, blocks=3):
    # divide the input image into NXN blocks
    (h,w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks+1, dtype='int')
    ySteps = np.linspace(0, h, blocks+1, dtype='int')

    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            startX = xSteps[j-1]
            startY = ySteps[i-1]
            endX = xSteps[j]
            endY = ySteps[i]

            # extracting ROI using numpy array slicing
            roi = image[startY:endY, startX:endX]

            # compute the mean of ROI for each channel
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]

            # draw a rectangle with mean values in the face image
            cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)

    return image


# path = r'C:\Users\aishw\PycharmProjects\faceblur\examples\Johnny.jpg'
# image = cv2.imread(path)
# cv2.imshow('Image', image)
# blurred_image = anonymize_face_pixelate(image)
# cv2.imshow('Blurred image', blurred_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


