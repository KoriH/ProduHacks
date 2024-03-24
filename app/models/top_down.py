import numpy as np
import os
import cv2
from tracker import *

os.chdir('C:/Users/ishaa/Downloads')

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
tracker = EuclideanDistTracker()
vidObj = cv2.VideoCapture('topdown.mp4') 

frame_id = 0
centroids = {}
velocities = {}
boxes = {}
frames = {}

def compute_velocity(tracker_id, centroid_x, centroid_y):
    # Recall previous centroid location of each object
    prev_x, prev_y = centroids[tracker_id]
    
    # Convert time between last occurence and current occurence to seconds
    seconds = frames[tracker_id] / 30
    
    # Compute x and y velocities respectively
    velocityx = centroid_x - prev_x / seconds
    velocityy = centroid_y - prev_y / seconds
    
    # Apply Pythagorean theorem to compute velocity
    velocity = np.sqrt(velocityx ** 2 + velocityy ** 2)
    velocities[tracker_id] = velocity

while vidObj.isOpened():
    # Read a frame from the video
    success, frame = vidObj.read()
    
    if success:
        # Increment frame count
        frame_id += 1
        
        # Resize the frame
        original_length = frame.shape[1] * 2
        original_height = frame.shape[0] * 2
        old_frame = cv2.resize(frame, (original_length, original_height))
        
        # Crop the frame for image detection
        roi = frame[40:320, 50:610]
        roi = cv2.resize(roi, (roi.shape[1] * 2, roi.shape[0] * 2))
        
        # Apply the object detector to the cropped frame
        mask = object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Isolated detections in the frame based on size criteria
        detections = []
        for cnt in contours:
            # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 50 and area < 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])

        # Update tracker with detections
        boxes_ids = tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            
            # Calculate centroid
            cx = x + w // 2
            cy = y + h // 2
            
            # Compute velocity
            compute_velocity(id, cx, cy)
            
            # Annotate frame
            cv2.putText(old_frame, str(velocities[box_id]), (x+100, y + 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.rectangle(old_frame, (x+100, y+80), (x + w + 100, y + h+80), (0, 255, 0), 3)

        # Display the annotated frame
        cv2.imshow("OpenCV Inference", old_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
vidObj.release()
cv2.destroyAllWindows()