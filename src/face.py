import numpy as np
import cv2
from keras.models import load_model
import keras
import tensorflow as tf
from PIL import Image
emotion = ['happy','neutral','sad']

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
classifier = load_model('mybabyclassifier_model.h5')
cap = cv2.VideoCapture('baby.mp4')

while (True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   # frame=cv2.resize(frame,(500,500))
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        # print(x,y,w,h)
        roi_gray = gray[y:y + h][x:x + w]

        roi_color = frame[y:y + h][x:x + w]

        roi_gray = cv2.resize(roi_color, (300, 300), interpolation=cv2.INTER_AREA)

        color = (255, 0, 0)  # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        roi = roi_gray
        #roi = tf.keras.preprocessing.image.array_to_img(roi)
        roi = np.expand_dims(roi, axis=0)
        prediction = classifier.predict(roi)[0]
        label = emotion[prediction.argmax()]

        print(label )

        label_position = (x,y -10)
        cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)


    cv2.imshow('frame', frame)
    if cv2.waitKey(20)&0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
