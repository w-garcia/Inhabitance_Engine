# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

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

rectsPersist = None
pickPersist = None

while(True):
    # load the image and resize it to (1) reduce detection time
    # and (2) improve detection accuracy

    retval, im = video_capture.read()
    image = im.copy()
    count += 1
    orig = image.copy()

    # draw the previous original bounding boxes
    if rectsPersist != None:
        for (x, y, w, h) in rectsPersist:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # draw the previous final bounding boxes
    if pickPersist != None:
        for (xA, yA, xB, yB) in pickPersist:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    if not count % 1:
        count = 0
        #image = cv2.imread(imagePath)
        image = imutils.resize(image, width=min(400, image.shape[1]))


        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
            padding=(8, 8), scale=1.05)

        rectsPersist = rects

        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        pickPersist = pick

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # show some information on the number of bounding boxes
        filename = "webcam"
        print("[INFO] {}: {} original boxes, {} after suppression".format(
            filename, len(rects), len(pick)))

    # show the output images
    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    #cv2.waitKey()

video_capture.release()
cv2.destroyAllWindows()