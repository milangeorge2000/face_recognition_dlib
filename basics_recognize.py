import cv2
import face_recognition
import numpy as np


imgelon = face_recognition.load_image_file("Image/elon.jpg")
imgelon  =  cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file("Image/Elon-Musk1.jpg")
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)


face_locations = face_recognition.face_locations(imgelon)[0]
face_encodings =  face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon,(face_locations[3],face_locations[0]),(face_locations[1],face_locations[2]),(255,0,255),2)


face_locations_test = face_recognition.face_locations(imgtest)[0]
face_encodings_test =  face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(face_locations_test[3],face_locations_test[0]),(face_locations_test[1],face_locations_test[2]),(255,0,255),2)


results = face_recognition.compare_faces([face_encodings],face_encodings_test)
distance = face_recognition.face_distance([face_encodings],face_encodings_test)

print(results)
print(distance)
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
import numpy as np
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2

cv2.imshow('Elon Musk',imgelon)
cv2.putText(imgtest, f'Distance is {np.round(distance[0],4)}', org, font, fontScale, color, thickness, cv2.LINE_AA)
cv2.imshow('Elon Test',imgtest)
cv2.waitKey(0)