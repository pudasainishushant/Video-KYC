import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy import spatial

import os  
import pytesseract

import datetime
import re
import difflib

import json



detector = MTCNN()
embedder = FaceNet()



def get_face(img):
    try:
    	face = detector.detect_faces(img)
    	face = face[0]['box']
    	img = img[face[1]:face[1]+face[3], face[0]:face[0]+ face[2]]
    	return img
    except:
        return img


def get_embeddings(img):
	img = cv2.resize(img, (160,160))
	img = img.reshape(1, 160,160,3)
	embeddings = embedder.embeddings(img)
	return embeddings


def get_cosine_similarity_score(emb1, emb2):
	result = 1 - spatial.distance.cosine(emb1, emb2)
	return result


def get_info(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(image)
    text = [line for line in text.split('\n') if line.strip() != '']
#    print(text)

    info = {}
    dis  = []
    ward = []

    for i in text:
        cnText = i.split(' ')
        cnText = [re.sub("[$&+,:;=?@#|'<>.^*()%!-]", '', c) for c in cnText]

        # for citizenship number and gender
        r = difflib.SequenceMatcher(None, 'Citizenship Certificate No. Sex Male Female Third',i).ratio()
#        print(r, type(r), i)
        if r >= 0.5:
            try:
                cn = [c for c in cnText if re.search("[0-9]", c)]
                cn = '-'.join(cn)
                if cn:
                    info['Citizenship Number'] = cn

                if difflib.SequenceMatcher(None, 'Male', cnText[-1]).ratio() > 0.75:
                    gender = "Male"
                elif difflib.SequenceMatcher(None, 'Female', cnText[-1]).ratio() > 0.75:
                    gender = "Female"
                elif difflib.SequenceMatcher(None, 'Third', cnText[-1]).ratio() > 0.75:
                    gender = "Third"

                info["Gender"] = gender
            except:
 #               print("Searching for citizenship number!!!!!")
                pass

        # for person name
        if re.findall(re.compile(r'Full Name'),i):
            try:
                cnText.remove('.')
            except:
                pass

            name = ' '.join(cnText[2:])
            info['Name'] = name.strip()    

        # for DOB
        mInitial = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
        if re.findall(re.compile(r'Date of Birth'), i):
#            print(i)
            year_and_day = [''.join(re.findall(re.compile(r"[0-9]"), y)) for y in cnText]
            year = [y for y in year_and_day if len(y) >= 4][0]

            day = [y for y in year_and_day if 1<= len(y) <= 2][0]

            month =[''.join(dict.fromkeys(re.findall(re.compile(f"{[','.join(mInitial)]}"), y))) for y in cnText]
            month = [m for m in month if m in mInitial]
            month = datetime.datetime.strptime(month[0], "%b").month

            dob = datetime.datetime(int(year), int(month), int(day)).strftime('%Y-%m-%d')
            info["Date of Birth (AD)"] = dob


        # municipality and ward number
        if re.findall("Municipality|Ward", i):
            wardNumber = [''.join(re.findall(re.compile(r"[0-9]"), y)) for y in cnText]
            wardNumber = [y for y in wardNumber if 2 >= len(y) >=1][0]
            ward.append(wardNumber)


    dis.clear()

    info["Birth Place (Ward No.)"] = ward[0]
    info["Permanent Place (Ward No.)"] = ward[-1]
    ward.clear()

    return info



if __name__ == '__main__':
	image_path = r"C:\Users\aarya\Downloads\ct-back.jpg"
	image = cv2.imread(image_path)
#	embeddings = get_embeddings(image)
#	print(len(embeddings))
#	print(embeddings[0].shape)
#	print(embeddings)
	info = get_info(image)
	print(info)