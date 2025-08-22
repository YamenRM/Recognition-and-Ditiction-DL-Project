import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
import cv2
from PIL import Image


# Initialize MTCNN globally 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

# FaceNet preprocessing transform
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1,1] range
])

def face_preprocessing(frame):
    # Load and convert image
    image = cv2.imread(frame)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, _ = mtcnn.detect(image_rgb)
    face_tensors = []

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face_crop = image_rgb[y1:y2, x1:x2]  # Crop face
            pil_face = Image.fromarray(face_crop)

            # Preprocess cropped face
            tensor_face = transform(pil_face)
            face_tensors.append(tensor_face)

            # Draw bounding box for visualization
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return face_tensors, boxes  # list of faces + boxes
