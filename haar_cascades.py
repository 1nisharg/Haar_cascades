import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#CascadeClassifer helps to load .XML files and features within it.


image = cv2.imread("Modi.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = face_classifier.detectMultiScale(gray, 1.3, 5)
#detectMultiScale will help to detect the features of our face or exact location of our face.

if faces is ():
  print("No faces found")

for (x,y,w,h) in faces:
  cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2)
  cv2.imshow("Face Detection", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# x -> X coordinate, y -> Y coordinate, w -> width, h -> height, (127,0,255)-> colors

