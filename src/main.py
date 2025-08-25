# load the model once  then recomment it

#from recognition import train_face_recognizer
#train_face_recognizer(data_dir="DATA", model_path="models/face_svm.pkl")



import cv2
from recognition import recognize_faces
from ditiction import object_dit 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # object detection
    frame, objects = object_dit(frame)

    # face recognition
    frame, names = recognize_faces(frame, model_path="models/face_svm.pkl")

    cv2.imshow("Objects + Faces + Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()