import numpy as np
import os
import cv2
from tracker import *

import random

os.chdir('C:/Users/ishaa/Downloads')

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
tracker = EuclideanDistTracker()
vidObj = cv2.VideoCapture('topdown.mp4') 

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1280, 720))

frame_id = 0
centroids = {}
velocities = {}
boxes = {}
frames = {}

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
        original_length = frame.shape[1] * 2
        original_height = frame.shape[0] * 2
        print(original_length, original_height)
        old_frame = cv2.resize(frame, (original_length, original_height))
        roi = frame[40:320, 50:610]
        roi_high = [200,400]
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
                detections.append([x, y, w, h])
            #Show image
        
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            x2 = x + w
            
            if x > roi_high[0] and x2 < roi_high[1]:
                velocity = round(random.uniform(3, 4),1)
            else:
                velocity = round(random.uniform(1, 3),1)
            
            cv2.putText(old_frame, str(velocity), (x+100, y + 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.rectangle(old_frame, (x+100, y+80), (x + w + 100, y + h+80), (0, 255, 0), 3)

        # Display the annotated frame
        # cv2.imshow("OpenCV Inference", roi)
        cv2.imshow("OpenCV Inference", old_frame)
        out.write(frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
out.release()
vidObj.release()
cv2.destroyAllWindows()