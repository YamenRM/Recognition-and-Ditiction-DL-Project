from ultralytics import YOLO
import cv2

def detect_objects(image_path):
    # Load the YOLOv8 model
    model = YOLO('models/yolov8n.pt')

    # Load the image 
    image = cv2.imread(image_path)

    # Perform object detection
    results = model(image)

    # Draw bounding boxes on the image
    annotated_image = results[0].plot()

    # show the name of the detected object
    for box in results[0].boxes:
     cls_id = int(box.cls)
     label = model.names[cls_id]
     conf = box.conf.item()
     print(f"Detected: {label} ({conf:.2f})")
    # use the live camera feed for live detection
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        results = model(frame)
        
        # Draw bounding boxes on the frame
        annotated_frame = results[0].plot()
        cv2.imshow('Object Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


