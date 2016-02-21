# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
#from __future__ import print_function
from imutils.object_detection import non_max_suppression
from time import sleep
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
from datetime import datetime

def beginVideoProcess(im):
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy
    image = im.copy()
    orig = image.copy()

    #image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(400, image.shape[1]))

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

#video_capture = cv2.VideoCapture(1)
camera = PiCamera()
rawCapture = PiRGBArray(camera)

count = 0
# loop over the image paths
#for imagePath in paths.list_images(args["images"]):

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    im_temp = frame.array
    
    # show the frame
    if im_temp == None:
        print('Got blank picture. Try a different hardware port')
        break
    im_temp = imutils.resize(im_temp, width=min(400, im_temp.shape[1]))
    cv2.imshow("Menu", im_temp) 
 
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    if cv2.waitKey(5) & 0xFF == ord('c'):
        print("Starting")
        for i in range(0, 10):
            sleep(1)
            count += beginVideoProcess(im_temp)
        print("[UPDATE]: {}: Average of 10 processes: {}".format(datetime.now(), count/10))


#video_capture.release()
cv2.destroyAllWindows()

