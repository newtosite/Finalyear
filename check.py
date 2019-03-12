import cv2,os
import numpy as np
from PIL import Image 
import pickle
import sqlite3
import webbrowser
import time
import msvcrt as m
from tkinter import *
import mysql.connector
from mysql.connector import Error
root=Tk()
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read('Rec_auth/traingdata.yml')

cascadepath="haarcascade_frontalface_default.xml"

facecascade=cv2.CascadeClassifier(cascadepath);

path='sum_samples'
def enable():
	
	root.title("WEAR HELMET")
	#root.insert(END, "Just a text Widget\nin two lines\n")
	w = 300 # width for the Tk root
	h = 200 # height for the Tk root

	# get screen width and height
	ws = root.winfo_screenwidth() # width of the screen
	hs = root.winfo_screenheight() # height of the screen

	# calculate x and y coordinates for the Tk root window
	x = (ws/2) - (w/2)
	y = (hs/2) - (h/2)

	# set the dimensions of the screen 
	# and where it is placed
	root.geometry('%dx%d+%d+%d' % (w, h, x, y))

	#root.geometry("1200*1024")
	app=Frame(root)
	app.grid()
	button1=Button(app,text="OK",height=2,width=10, command=helmetCheck)
	button1.grid()
	button2=Button(app,text="Disable",height=2,width=10, command=scan1)
	button2.grid()
	root.mainloop()
def telematic():
	chrome_path='C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'
	webbrowser.register('chrome', None,webbrowser.BackgroundBrowser(chrome_path),1)
	webbrowser.get('chrome').open_new_tab('http://localhost/kpit/telematic_main.html')
def scan1():
	root.destroy()
	count=1
	try:
		files=open("disabled.txt","r")
		temp=files.read()
		temp=temp+count
		#temp=count+1
		
		files=open("count.txt","w")
		files.write("%f"%temp)
		files.close()
	except:
		files.close()
		files=open("disabled.txt","w")
		files.write("%f"%count)
		files.close()
		print(count,"\n")
	
	scan()
def scan():
		cam = cv2.VideoCapture(0)
		font = cv2.FONT_HERSHEY_SIMPLEX
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
					
		from PIL import Image 
		image = Image.open('license.jpg')	
		greyscale_image = image.convert('L')
		greyscale_image.save('licenceGrey.jpg');	
		img = cv2.imread('licenceGrey.jpg')
		b,g,r = cv2.split(img)           # get b,g,r
		rgb_img = cv2.merge([r,g,b])  
		dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
		b,g,r = cv2.split(dst)           # get b,g,r
		rgb_dst = cv2.merge([r,g,b])    # switch it to rgb
		cv2.imwrite('licenceDenoised1.jpg',rgb_dst)
		try:
			from PIL import Image
		except ImportError:
			import Image
		import pytesseract
		licence =  ['licencing','dl','valid','india','mcwg','lmv','name','till','throughout','16(2)','bg','doi','cov','dob']
		rc = ['reg','chassis','engine','mfr','class','registering','colour','ownername','s/w/d','model','body','wheel','base','23A','mfg','fuel','tax','seating','cc']
		rcCounter = 0
		licenceCounter = 0
		foo = pytesseract.image_to_string(Image.open('C:\\Users\\pc\\python programs\\ocr\\licenceGrey.jpg'))
		x = open("licence1.txt","w")
		for word in foo:
			if ( word.isalpha() or word.isdigit() or word == '/' or word == '(' or word == ')'): #if(word == ':' or word == '.' or word == '!' or word == '/' or word == '-' or word == '[' or word == '>' or word == '@' or word == '=' ):
				x.write("%s" %word)
			elif(word == " "):
				x.write("\n")
			else:
				x.write("\n")
				#x = open("licence.txt","w")
				#x.write("%s" % foo)
		x.close()
		file = open("licence1.txt","r")
		for line in file:
			line = line.strip()
			line = line.lower()
			if line in licence:
				licenceCounter = licenceCounter+1
				#print(licenceCounter)
			if line in rc :
				rcCounter = rcCounter+1
				#print(" rc ",rcCounter)
		if licenceCounter > rcCounter :
			print ( " \nthe document is a licence ")
			chrome_path='C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'
			webbrowser.register('chrome', None,webbrowser.BackgroundBrowser(chrome_path),1)
			webbrowser.get('chrome').open_new_tab('http://localhost/kpit/telematic_main.html')
		elif licenceCounter < rcCounter:
			print ( " the document is a rc ")
		else:
			print(" unable to identify ")
		file.close()
		exit()
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
	#print("\n wear helmet...\n")
	root.destroy()
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
			id,cof = rec.predict(gray[y:y+h+100, x:x+w+100])
			cv2.rectangle(im,(x,y),(x+w,y+h), (225,0,0),2)
			cv2.putText(im,"Checking for helmet", (35,35),font,scale,color,lineType=cv2.LINE_AA)
			profile = getprofile_of_helmet(id)
			#face_resize = cv2.resize(faces, (100,100))
			prediction = rec.predict(gray[y:y+h+30, x:x+w+100])
			helmet=0
			#print(prediction[1])
			#print("\n")
			if prediction[1]>70:
				cv2.putText(im,"No helmet", (x,y+h+30), font,scale,color,lineType=cv2.LINE_AA)
				no+=1
				if no>60:
					cam.release()
					cv2.destroyAllWindows() 
					
					exit()
				
			else:
				cv2.putText(im, str(profile[1]), (x,y+h+100), font,scale,color,lineType=cv2.LINE_AA)
				yes+=1
				helmet+=1
				if yes>60:
					cv2.imshow('im',im)
					cam.release()
					cv2.destroyAllWindows()
					scan()
					time.sleep(2)
					
					
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
		#print(prediction[1])
		#print("\n")
		if prediction[1]>50:

			cv2.putText(im,"UNKNOWN", (x,y+h+30), font,scale,color,lineType=cv2.LINE_AA)
			no+=1
			if no>60:
				cam.release()
				cv2.destroyAllWindows() 
				#helmetCheck()
				enable()
				exit()
		else:
			cv2.putText(im, str(profile[1]), (x,y+h+30), font,scale,color,lineType=cv2.LINE_AA)
			yes+=1
			if yes>35:
				cam.release()
				cv2.destroyAllWindows()
				#helmetCheck()
				enable()
				exit()
	cv2.imshow('im',im)
	if cv2.waitKey(1)== ord('q'):
			break;
cam.release()
cv2.destroyAllWindows() 




	