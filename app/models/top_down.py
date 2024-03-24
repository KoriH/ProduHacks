import numpy as np
import os
import cv2
from tracker import *

os.chdir('C:/Users/ishaa/Downloads')

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
tracker = EuclideanDistTracker()
vidObj = cv2.VideoCapture('test1.mp4') 

light_white = (0, 0, 200)
dark_white = (145, 60, 255)


frame_id = 0
centroids = {}
velocities = {}
boxes = {}
frames = {}

# turn up contrast

def compute_velocity(tracker_id, centroid_x, centroid_y):
    prev_x, prev_y = centroids[tracker_id]
    velocity = centroid_x - prev_x / frames[tracker_id]
    # conversion of frames to seconds
    velocities[tracker_id] = velocity

def annotate_frame(frame, x1, y1, x2, y2, tracker_id):
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    frame = cv2.putText(frame, f"{velocities[tracker_id]}", (x2+7, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
    return frame

# mock random velocity values based on regions of interest

while vidObj.isOpened():
    # Read a frame from the video
    success, frame = vidObj.read()
    
    #resize frames to be larger
    
    if success:
        frame_id += 1
        
        roi = frame[40:320, 50:610]
        roi = cv2.resize(roi, (roi.shape[1] * 2, roi.shape[0] * 2))
        
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 50 and area < 1000:
                # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(cnt)
                # shift and write back onto main image
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
                detections.append([x, y, w, h])
            #Show image
        
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # reverse y comparisons
        for box in boxes:
            pass
            # help quant tree????
            
        # Display the annotated frame
        cv2.imshow("OpenCV Inference", roi)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
vidObj.release()
cv2.destroyAllWindows()