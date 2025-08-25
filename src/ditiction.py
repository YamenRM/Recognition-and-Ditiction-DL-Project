from ultralytics import YOLO
import cv2
import numpy as np

# load the yolo model
model= YOLO('models/yolov8n.pt')

# object ditiction func
def object_dit(frame):
    results = model(frame , conf=0.3 , verbose=False)[0]
     # make a list for the diticted items
    detections = []

    # draw boxes and labels
    annotated = results.plot()
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box.tolist())
        label = results.names[int(cls)]
        detections.append((label, float(conf), (x1, y1, x2, y2)))
    return annotated, detections

    
    


