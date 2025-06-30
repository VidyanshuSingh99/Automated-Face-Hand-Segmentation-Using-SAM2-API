
from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8n-face or hand model
face_model = YOLO('yolov8-face.pt')  
hand_model = YOLO('yolov8n-hand.pt')       

def detect_objects(image_path):
    image = cv2.imread(image_path)

    # Detect faces
    face_results = face_model(image)
    hands_results = hand_model(image)

    boxes = []

   
    for result in face_results:
        for box in result.boxes.xyxy:
            boxes.append(("face", box.cpu().numpy()))

    for result in hands_results:
        for box in result.boxes.xyxy:
            boxes.append(("hand", box.cpu().numpy()))

    return image, boxes
