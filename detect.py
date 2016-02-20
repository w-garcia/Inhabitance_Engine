# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from time import sleep
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from datetime import datetime

def beginVideoProcess():
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy

    retval, im = video_capture.read()
    image = im.copy()
    orig = image.copy()

    #image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(900, image.shape[1]))

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
        padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show some information on the number of bounding boxes
    filename = "webcam"
    print("[INFO] {}: {}".format(filename, len(pick)))
    # show the output images
    cv2.imshow("After NMS", image)

    return len(pick)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

video_capture = cv2.VideoCapture(0)
count = 0
# loop over the image paths
#for imagePath in paths.list_images(args["images"]):


while(True):
    rv, im_temp = video_capture.read()
    cv2.imshow("Menu", im_temp)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    if cv2.waitKey(5) & 0xFF == ord('c'):
        print("Starting")
        for i in range(0, 10):
            sleep(1)
            count += beginVideoProcess()
            print("[STATUS] Finished one process, found {}".format(count))
        print("[UPDATE]: {}: Average of 10 processes: {}".format(datetime.now(), count/10))


video_capture.release()
cv2.destroyAllWindows()

