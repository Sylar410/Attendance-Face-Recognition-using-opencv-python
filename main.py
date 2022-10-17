import cv2
import numpy as np
import face_recognition

imgJay = face_recognition.load_image_file('ImageBasic/jayesh.jpg')
imgJay = cv2.cvtColor(imgJay,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImageBasic/jayesh test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLocation = face_recognition.face_locations(imgJay)[0]
encodeJay = face_recognition.face_encodings(imgJay)[0]
cv2.rectangle(imgJay,(faceLocation[3],faceLocation[0]),(faceLocation[1],faceLocation[2]),(255,0,255),2)

faceLocationTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocationTest[3],faceLocationTest[0]),(faceLocationTest[1],faceLocationTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeJay],encodeTest)
faceDistance = face_recognition.face_distance([encodeJay],encodeTest)
print(results,faceDistance)
cv2.putText(imgTest,f'{results} {round(faceDistance[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('jayesh',imgJay)
cv2.imshow('jayesh test', imgTest)
cv2.waitKey(0)
