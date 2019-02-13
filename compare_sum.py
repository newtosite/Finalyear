import cv2,os
import numpy as np
from PIL import Image 
import pickle
import sqlite3
import webbrowser
import time
import msvcrt as m

rec=cv2.face.LBPHFaceRecognizer_create()
rec.read('Rec_auth/traingdata.yml')

cascadepath="haarcascade_frontalface_default.xml"

facecascade=cv2.CascadeClassifier(cascadepath);

path='sum_samples'
def getprofile(id):
	connection = sqlite3.connect("summarize.db")
	cmd = " SELECT * FROM Authentication WHERE Id ="+str(id)
	cursor = connection.execute(cmd)
	profile = None
	for row in cursor :
		profile = row
	connection.close()
	return profile
	
def getprofile_of_helmet(id):
	connection = sqlite3.connect("helmet.db")
	cmd = " SELECT * FROM helmetdetect WHERE Id ="+str(id)
	cursor = connection.execute(cmd)
	profile = None
	for row in cursor :
		profile = row
	connection.close()
	return profile
	
	
def helmetCheck():
	print("\n wear helmet...\n")
	rec=cv2.face.LBPHFaceRecognizer_create()
	rec.read('Rec_helmet/traingdata.yml')

	cascadepath="haarcascade_frontalface_default.xml"

	facecascade=cv2.CascadeClassifier(cascadepath);

	path='helmet_samples'
	cam = cv2.VideoCapture(0)
	font = cv2.FONT_HERSHEY_SIMPLEX
	scale=1.2
	color=(0,0,255)
	yes=0
	no=0
	while True:
		ret,im=cam.read()
		gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
		faces = facecascade.detectMultiScale(gray,scaleFactor = 1.2, minNeighbors = 5, minSize=(100,100), flags = cv2.CASCADE_SCALE_IMAGE)
		for (x,y,w,h) in faces : 
			id,cof = rec.predict(gray[y:y+h, x:x+w])
			cv2.rectangle(im,(x,y),(x+w,y+h), (225,0,0),2)
			cv2.putText(im,"Checking for helmet", (35,35),font,scale,color,lineType=cv2.LINE_AA)
			profile = getprofile_of_helmet(id)
			#face_resize = cv2.resize(faces, (100,100))
			prediction = rec.predict(gray[y:y+h, x:x+w])
			helmet=0
			
			if prediction[1]>50:
				cv2.putText(im,"No helmet", (x,y+h+30), font,scale,color,lineType=cv2.LINE_AA)
				no+=1
				if no>60:
					cam.release()
					cv2.destroyAllWindows() 
					exit()
				
			else:
				cv2.putText(im, str(profile[1]), (x,y+h+30), font,scale,color,lineType=cv2.LINE_AA)
				yes+=1
				helmet+=1
				if yes>60:
					cv2.imshow('im',im)
					cam.release()
					cv2.destroyAllWindows()
					
					
					time.sleep(2)
					print("Scan license...")
					
					cam = cv2.VideoCapture(0)
					ch='continue';
					while ch!='stop':
						ret,im=cam.read()
						cv2.imshow('im',im)
						if cv2.waitKey(1)== ord('q'): 
							cv2.imwrite("license.jpg", im)
							ch='stop'
							break;
					
							
						
					cam.release()
					cv2.destroyAllWindows()
					
					chrome_path='C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'
					webbrowser.register('chrome', None,webbrowser.BackgroundBrowser(chrome_path),1)
					webbrowser.get('chrome').open_new_tab('http://localhost/kpit/telematic_main.html')
					
		cv2.imshow('im',im)
		if cv2.waitKey(1)== ord('q'):
			break;
	cam.release()
	cv2.destroyAllWindows() 
	
	
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
scale=1.2
color=(0,0,255)
yes=0
no=0
while True:
	ret,im=cam.read()
	gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	faces = facecascade.detectMultiScale(gray,scaleFactor = 1.2, minNeighbors = 5, minSize=(100,100), flags = cv2.CASCADE_SCALE_IMAGE)
	for (x,y,w,h) in faces : 
		id,cof = rec.predict(gray[y:y+h, x:x+w])
		cv2.rectangle(im,(x,y),(x+w,y+h), (225,0,0),2)
		cv2.putText(im,"Authenticating user", (35,35),font,scale,color,lineType=cv2.LINE_AA)
		profile = getprofile(id)
		#face_resize = cv2.resize(faces, (100,100))
		prediction = rec.predict(gray[y:y+h, x:x+w])
		
		if prediction[1]>50:

			cv2.putText(im,"UNKNOWN", (x,y+h+30), font,scale,color,lineType=cv2.LINE_AA)
			no+=1
			if no>60:
				cam.release()
				cv2.destroyAllWindows() 
				helmetCheck()
				exit()
		else:
			cv2.putText(im, str(profile[1]), (x,y+h+30), font,scale,color,lineType=cv2.LINE_AA)
			yes+=1
			if yes>35:
				cam.release()
				cv2.destroyAllWindows()
				helmetCheck()
				exit()
	cv2.imshow('im',im)
	if cv2.waitKey(1)== ord('q'):
			break;
cam.release()
cv2.destroyAllWindows() 
	

	