from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
from decord import VideoReader, cpu

os.chdir('C:/Users/ishaa/Downloads')

vr = VideoReader('test.mp4', ctx=cpu(0))
print('video frames:', len(vr))
for i in range(len(vr)):
    frames = vr.get_batch(range(i*BATCH_SIZE, (i+1)*BATCH_SIZE)).asnumpy()
    print(frames.shape)
    print(frames.dtype)
    print(frames.context)

model = YOLO("yolov8n.pt")

box_annotator = sv.BoundingBoxAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    return box_annotator.annotate(frame.copy(), detections=detections)

sv.process_video(
    source_path="test.mp4",
    target_path="result.mp4",
    callback=callback
)