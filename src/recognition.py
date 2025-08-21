from utils import FaceRecognitioon
import torch
from facenet_pytorch import  InceptionResnetV1
import cv2 
from sklearn.svm import SVC
import numpy as np
from PIL import Image

# Initialize Face Recognition globally
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


