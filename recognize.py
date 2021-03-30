import os
import face_recognition
import numpy as np
import cv2
from datetime import datetime

path = "Image"
images = []
names = []

files = os.listdir(path)

print(files)

## To read each image and name of the image and save it in the form of list
for image in files:
    cur_img = cv2.imread(f'{path}/{image}')
    images.append(cur_img)
    names.append(image.split('.')[0])


def find_encodings(images):
    image_encodings = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)[0]
        image_encodings.append(encodings)
    return image_encodings

def markAttendance(name):
 with open('attendence.csv','r+') as f:
  myDataList = f.readlines()
  nameList = []
  for line in myDataList:
   entry = line.split(',')
   nameList.append(entry[0])
  if name not in nameList:
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    f.writelines(f'\n{name},{dtString}')


known_encodings = find_encodings(images)
print("Encoding Completed..")


cap = cv2.VideoCapture(0)

while True:
    _,img = cap.read()
    resized_image = cv2.resize(img,(0,0),None,0.25,0.25)
    resized_image = cv2.cvtColor(resized_image,cv2.COLOR_BGR2RGB)

    face_loc_cur = face_recognition.face_locations(resized_image)
    face_encodings_cur = face_recognition.face_encodings(resized_image,face_loc_cur)

    for encoded_face,face_locations in zip(face_encodings_cur,face_loc_cur):
        matches = face_recognition.compare_faces(known_encodings,encoded_face)
        distance = face_recognition.face_distance(known_encodings,encoded_face)
        print(distance)
        match_index = np.argmin(distance)

        if matches[match_index]:
            name = names[match_index].upper()
            print(name)
            x1,y1,x2,y2 = face_locations
            x1,y1,x2,y2 = x1*4,y1*4,x2*4,y2*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-55),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



