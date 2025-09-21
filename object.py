import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os  # to avoid hardcoding the file
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
def face_detected(img): 
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    for(x,y,w,h) in face_rects : 
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
    return face_img  
def detect_eyes(img):
    eye_img = img.copy()
    eye_rect = eye_cascade.detectMultiScale(eye_img, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in eye_rect:
        cv2.rectangle(eye_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
    return eye_img 
path = "Screenshot 2025-09-20 222541.png"
img= cv2.imread(path) 
img_cpy1= img.copy()
img_cpy2= img.copy()
img_cpy3= img.copy()
if img is None:
    print("could not print")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
face = face_detected(img_cpy1)
plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
plt.show ()
eye = detect_eyes(img_cpy2)
plt.imshow(cv2.cvtColor(eye, cv2.COLOR_BGR2RGB))
plt.show()

