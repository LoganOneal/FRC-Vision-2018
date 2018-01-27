# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

from collections import deque
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
hsvLower = (28, 79, 103)
hsvUpper = (37, 255, 255)
pts = deque(maxlen=args["buffer"])

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('out.avi', fourcc, 20.0, (640,480))

# if no video source use /video0
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# else get video file
else:
    camera = cv2.VideoCapture(args["video"])


while True:
    (grabbed, origin) = camera.read()
    (grabbed, frame) = camera.read()
    frame=cv2.flip(frame,1)



    #this ends video at last frame
    if args.get("video") and not grabbed:
        break

    # frame: resize, bur, HSV
    origin = imutils.resize(frame, width=600)
    frame = imutils.resize(frame, width=600)
    frame = cv2.blur(frame, (12, 12))
    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV range filter then erode(get rid of small points) and dilate (this basically makes the mask less strict)
    mask = cv2.inRange(hsv, hsvLower, hsvUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ct = 0
        #for c in filter(lambda x: cv2.contourArea(x) > 15, cnts):
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if (radius > 10):
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.rectangle(origin ,(int(x)-int(radius), int(y)+int(radius)),(int(x)+int(radius), int(y)-int(radius)),(0,255,0),3)
            cv2.putText(origin,'LargestContour ' + str(round(x,1)) + ' ' + str(round(y,1)),(int(x)-int(radius), int(y)-int(radius)), cv2.FONT_HERSHEY_PLAIN, 2*(int(radius)/45),(255,255,255),2,cv2.LINE_AA)

            #cv.rectangle(frame , ((int(x)- int(radius), (int(y)+int(radius)) , (1,1),(0,255,0),3)
            cv2.circle(origin, center, 5, (255 * (ct % 3 == 2), 255 * (ct % 3 == 1), 255 * (ct % 3 == 0)), -1)
        ct += 1

    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    
    # write to file
    out.write(frame)
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    cv2.imshow("Original", origin)


    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()