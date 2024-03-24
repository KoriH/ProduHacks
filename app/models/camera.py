from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
import cv2

os.chdir('/Users/kori0909/Downloads')
model = YOLO("yolov8n.pt")
tracker = sv.ByteTrack()
vidObj = cv2.VideoCapture('topdown.mp4') 

frame_id = 0
centroids = {}
velocities = {}
boxes = {}
frames = {}
times = {}

def compute_velocity(tracker_id, centroid_x, centroid_y, scale_factor):
    prev_x, prev_y = centroids.get(tracker_id, (centroid_x, centroid_y))
    frame_count = frames.get(tracker_id, 1)
    dx = (centroid_x - prev_x) * scale_factor
    dy = (centroid_y - prev_y) * scale_factor
    velocity = np.sqrt(dx**2 + dy**2) / frame_count * 30
    velocities[tracker_id] = round(velocity, 1)

def annotate_frame(frame, x1, y1, x2, y2, tracker_id):
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    frame = cv2.putText(frame, f"{velocities[tracker_id]}", (x2+7, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
    return frame

def collision_frame(frame, tracker_id, other_tracker_id):
    box1 = boxes[tracker_id]
    box2 = boxes[other_tracker_id]
    
    x1_1, y1_1, x2_1, y2_1 = map(int, box1)
    x1_2, y1_2, x2_2, y2_2 = map(int, box2)
    
    # Calculate the center of the first box
    label_x1 = int((x1_1 + x2_1) / 2)
    label_y1 = int((y1_1 + y2_1) / 2)
    
    # Calculate the center of the second box
    label_x2 = int((x1_2 + x2_2) / 2)
    label_y2 = int((y1_2 + y2_2) / 2)
    
    # Draw a red rectangle around the first box
    frame = cv2.rectangle(frame, (x1_1, y1_1), (x2_1, y2_1), (0, 0, 255), 2)
    
    # Draw a red rectangle around the second box
    frame = cv2.rectangle(frame, (x1_2, y1_2), (x2_2, y2_2), (0, 0, 255), 2)
    
    # Add a label to the first box
    frame = cv2.putText(frame, "Minor Collision Detected!", (label_x1, label_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add a label to the second box
    frame = cv2.putText(frame, "Minor Collision Detected!", (label_x2, label_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame

def detect_collision(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    if x2_1 < x1_2 or x2_2 < x1_1:
        return False

    if y2_1 < y1_2 or y2_2 < y1_1:
        return False

    return True

cap = cv2.VideoCapture(0)  
annotated_frame = None 
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    
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
            ret, frame = cap.read()
            if ret:
                frame_height, frame_width = frame.shape[:2]
            compute_velocity(tracker_id, centroid_x, centroid_y, scale_factor=1)
            centroids[tracker_id] = (centroid_x, centroid_y)
            annotated_frame = annotate_frame(uncropped, x1, y1, x2, y2, tracker_id)
            frames[tracker_id] = frame_id

            for other_tracker_id, other_box in boxes.items():
                if other_tracker_id != tracker_id and detect_collision(boxes[tracker_id], boxes[other_tracker_id]):
                    annotated_frame = collision_frame(uncropped, tracker_id, other_tracker_id)
                else:
                    annotated_frame = annotate_frame(uncropped, x1, y1, x2, y2, tracker_id)
        
        # reverse y comparisons
        for box in boxes:
            pass
            # help quant tree????
            # if centroids are within distance then check object border
            
        # Display the annotated frame
        if annotated_frame is not None:
            cv2.imshow("YOLOv8 Inference", annotated_frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
cap.release()
cv2.destroyAllWindows()