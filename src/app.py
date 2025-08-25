from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import cv2
from recognition import recognize_faces
from ditiction import object_dit
from utils import face_preprocessing
import joblib

app = FastAPI(title='smart ditiction')

#load the model
model=joblib.load('models/face_svm.pkl')

@app.get('/')
def home():
    return{"message": "AI Assistant API is running 🚀"}

@app.post('/predict') 
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Object detection
    frame, objects = object_dit(frame)

    # Face recognition
    frame, names = recognize_faces(frame)

    # Convert image to bytes to return
    _, img_encoded = cv2.imencode('.jpg', frame)
    result_image_bytes = img_encoded.tobytes()

    return {
        "detected_objects": objects,
        "recognized_faces": names
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)