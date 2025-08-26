# 🤖 Real-Time Face Recognition & Object Detection

## Description:
  Real-time  detects objects and recognizes faces using deep learning. Built with PyTorch, YOLOv8, and FaceNet, and deployed via FastAPI for easy API access. The app supports live webcam feeds and secure API access. 🚀

## ✨ Features

 - 🟢 Real-Time Object Detection using YOLOv8

 - 🟢 Face Detection using MTCNN

 - 🟢 Face Recognition with transfer learning on custom dataset via FaceNet

 - 💻 Deployed using **Render** 

## 📁 Project Structure
```
 Project/
  │
  ├─ src/
  │   ├─ ditiction.py         # Object & face detection functions
  │   ├─ recognition.py       # Face recognition functions
  │   ├─ utils.py             # Preprocessing utilities
  │   ├─ main.py              # Main script: camera feed, detection, recognition (for live cam rec)
  │   └─ app.py               # FastAPI deployment (images rec)
  │
  ├─ models/
  │   └─ face_svm.pkl         # Trained face recognition model 
  │
  ├─ DATA/                    # Folder with images for training face recognition(**need data** required 2+ persons and 10+ image for each with diffrenet angels and lightning)
  │   └─ Name1
  │       └─image1.jpg
  ├─ requirements.txt
  └─ README.md
```

## 🌐 try it live:
https://recognition-and-ditiction-api.onrender.com



## ⚡ Installation

### Clone the repository:
```
git clone https://github.com/YamenRM/Recognition-and-Ditiction-DL-Project.git
cd Recognition-and-Ditiction-DL-Project
```

### Create a virtual environment and activate it:
```
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Install dependencies:
```
pip install -r requirements.txt
```

## Set your API key 🔑:
```
# Windows PowerShell
$env:API_KEY="your-secret-key"

# Linux/Mac
export API_KEY="your-secret-key"
```
## 🏃 Usage
  -  Train Face Recognition Model (run once after you put your data and then recomment it)
```
     from recognition import train_face_recognizer
    train_face_recognizer(data_dir="DATA", model_path="models/face_svm.pkl")
```

2. Run Locally
```
python src/main.py
```


 - Opens webcam 📷

 - Detects objects 🛑 and recognizes faces 🙂 in real-time

 - Press q to quit ❌


## 🚀 Future Improvements

 - Multi-face recognition and tracking 👥

 - Voice command integration 🎤

 - Remote live streaming via web interface 🌐

 - GPU-enabled cloud deployment for faster inference ⚡

## Author ✨

 - YamenRM

 - 🌍 GAZA| UP 3rd year

 - Email: [yamenrafat132@gmail.com]

 - Stay Strong! 💪
