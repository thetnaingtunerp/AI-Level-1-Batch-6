from tensorflow.keras.models import load_model
import numpy as np
import cv2 
import cvlib as cv
import os
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('gender_detection.h5')

classes = ['man', 'woman']

webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    status,frame=webcam.read()
    face, confideance = cv.detect_face(frame)
    for idx,f in enumerate(face):
        (startx, starty) = f[0],f[1]
        (endx, endy) =f[2], f[3]
        cv2.rectangle(frame, (startx, starty), (endx, endy), (255,0,0),2)
        face_crop = np.copy(frame[starty:endy, startx:endx])
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1] <10):
            continue
        
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype('float')/255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        
        conf = model.predict(face_crop)[0]
        
        idx = np.argmax(conf)
        label = classes[idx]
        
        label = "{} : {:.2f}%".format(label, conf[idx]*100) 
        
        Y = starty - 10 if starty -10 > 10 else starty +10
        
        cv2.putText(frame, label, (startx, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,0,255), 2)
    cv2.imshow("MyFace", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
    
               
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
