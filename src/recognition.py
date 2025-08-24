import os
from utils import face_preprocessing
import torch
import joblib
from facenet_pytorch import  InceptionResnetV1
import cv2 
from sklearn.svm import SVC
import numpy as np
from PIL import Image


# Initialize Face Recognition globally
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# train the model on the data 
def train_face_recognizer(data_dir ='DATA' , model_path='models/face_svm.pkl') :
 embedings , labels = [] , []

 for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith(('.jpg', '.png')):
                continue
            img_path = os.path.join(person_dir, img_name)
            face_tensors, _ = face_preprocessing(cv2.imread(img_path))
            if not face_tensors:
                continue
             # get embedings
            with torch.no_grad():
                for t in face_tensors:
                     emb = model(t.unsqueeze(0).to(device))
                     embedings.append(emb.cpu().numpy()[0])
                     labels.append(person)

 embedings = np.array(embedings)       
 labels = np.array(labels)       

# loading the training model
 clf = SVC(kernel='linear', probability=True)
 clf.fit(embedings, labels) 

 # saving the model
 os.makedirs(os.path.dirname(model_path), exist_ok=True)
 joblib.dump(clf, model_path)
 print(f"[INFO] Model saved to {model_path}")
 return clf    


def recognize_faces(frame, model_path="models/face_svm.pkl", threshold=0.6):
    clf = joblib.load(model_path)
    face_tensors, boxes = face_preprocessing(frame)
    names = []

    if face_tensors and boxes is not None:
        with torch.no_grad():
            embeddings = torch.stack(face_tensors).to(device)
            embeddings = model(embeddings).cpu().numpy()

        for emb, box in zip(embeddings, boxes):
            probs = clf.predict_proba([emb])[0]
            pred_idx = np.argmax(probs)
            pred_name = clf.classes_[pred_idx]
            confidence = probs[pred_idx]
            if confidence < threshold:
                pred_name = "Unknown"
            names.append(pred_name)

            # draw name
            x1, y1, x2, y2 = map(int, box)
            cv2.putText(frame, f"{pred_name} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    return frame, names




