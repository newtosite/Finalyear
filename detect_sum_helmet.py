import cv2
import sqlite3
import os
import numpy as np
from PIL import Image
cam=cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def insertUpdate(Id,Name):
	conn=sqlite3.connect("helmet.db")
	cmd="SELECT * FROM helmetdetect WHERE id="+str(Id);
	cursor=conn.execute(cmd)
	isRecordExists=0
	for row in cursor:
		isRecordExists=1
	if(isRecordExists==1):
		cmd="UPDATE helmetdetect SET Name="+str(Name)+"WHERE id="+str(Id)
	else:
		cmd="INSERT INTO helmetdetect(id,Name) VALUES("+str(Id)+","+str(Name)+")"
	conn.execute(cmd)
	conn.commit()
	conn.close()
	
id=input("Enter id")
name=input("Enter name")
insertUpdate(id,name)
sampleName=0
while True:
	ret,im=cam.read()
	gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	faces = detector.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(100,100), flags=cv2.CASCADE_SCALE_IMAGE)
	for (x,y,w,h) in faces : 
		sampleName=sampleName+1
		cv2.imwrite("helmet_samples/user."+str(id)+"."+str(sampleName)+".jpg",gray[y:y+h,x:x+h])
		cv2.rectangle(im,(x,y),(x+w,y+h), (225,0,0),2)
	cv2.imshow('im',im)
	cv2.waitKey(100)
	if sampleName>50:
		cam.release()
		break;
recoginzer=cv2.face.LBPHFaceRecognizer_create()
path='helmet_samples'

def getImageWithId(path):
	imagepaths=[os.path.join(path,f) for f in os.listdir(path)]
	print(imagepaths)
	faces=[]
	ids=[]
	for imagePath in imagepaths:
		faceImg=Image.open(imagePath).convert('L');
		faceNp=np.array(faceImg,'uint8')
		ID=int(os.path.split(imagePath)[-1].split('.')[1])
		faces.append(faceNp)
		ids.append(ID)
		cv2.waitKey(10)
	return ids,faces
	
ids,faces=getImageWithId(path)
recoginzer.train(faces,np.array(ids))
recoginzer.save("Rec_helmet/traingdata.yml")
cv2.destroyAllWindows() 
		

		