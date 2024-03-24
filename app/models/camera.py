from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
import cv2

os.chdir('C:/Users/ishaa/Downloads')

model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
vidObj = cv2.VideoCapture('test.mp4') 

frame_id = 0
centroids = {}
velocities = {}
boxes = {}
frames = {}
times = {}

# calibration using time between occurences
# mock random velocity values based on regions of interest

def compute_velocity(tracker_id, centroid_x, centroid_y):
    prev_x, prev_y = centroids[tracker_id]
    # velocity = np.sqrt((centroid_x - prev_x) ** 2 + (centroid_y - prev_y) ** 2)
    velocity = centroid_x - prev_x / frames[tracker_id]
    # conversion of frames to seconds
    velocities[tracker_id] = round(velocity, 1)

# green instead of ruple
def annotate_frame(frame, x1, y1, x2, y2, tracker_id):
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    frame = cv2.putText(frame, f"{velocities[tracker_id]}", (x2+7, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    return frame

def collision_frame(frame, tracker_id1, tracker_id2):
    box1 = boxes[tracker_id1]
    box2 = boxes[tracker_id2]
    label_x = max(abs(box1[2] - box2[0]) / 2, abs(box1[0] - box2[2]) / 2)
    
    label_y = max(box1[3], box2[3]) + 10
    
    frame = cv2.putText(frame, "Collision Detected!", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    
    # different color + text at bottom (bold and red)
    
    return frame

while vidObj.isOpened():
    # Read a frame from the video
    success, frame = vidObj.read()
    
    if success:
        frame_id += 1
        uncropped = frame.copy()
        frame = frame[220:530, :]
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        
        for tracker_id, box in zip(detections.tracker_id, detections.xyxy):
            x1, y1, x2, y2 = box 
            y1, y2 = y1 + 220, y2 + 220
            boxes[tracker_id] = (x1, y1, x2, y2)
            
            centroid_x = (x1 + x2) / 2
            centroid_y = (y1 + y2) / 2
            if tracker_id in centroids:
                compute_velocity(tracker_id, centroid_x, centroid_y)
            else:
                velocities[tracker_id] = 0
            centroids[tracker_id] = (centroid_x, centroid_y)
            annotated_frame = annotate_frame(uncropped, x1, y1, x2, y2, tracker_id)
            frames[tracker_id] = frame_id
        
        # reverse y comparisons
        for box in boxes:
            pass
            # help quant tree????
            # if centroids are within distance then check object border
            # https://stackoverflow.com/questions/40795709/checking-whether-two-rectangles-overlap-in-python-using-two-bottom-left-corners 
            
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
vidObj.release()
cv2.destroyAllWindows()