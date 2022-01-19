import os, sys
import numpy as np
import cv2
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ocr.back_side_parse import get_info_back, get_string_similarity
from ocr.front_side_parse import get_info_citizenship

my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, my_absolute_dirpath + '/face')
from face import dnn_face_detection, find_face_similarity

sys.path.insert(0, my_absolute_dirpath + '/passive_liveness')
from passive_liveness import predict_liveness

from active_liveness import headpose_liveness, blink_liveness

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

def get_face(img):
	status = False
	face_box = dnn_face_detection.detect_face(img)
	if len(face_box) == 0:
		return status, "No Face Detected"
	elif len(face_box) >= 2:
		return status, "Multiple Faces Detected"
	elif len(face_box) == 1:
		box = face_box[0]
		x , y , w, h = box.astype('int')
		status = True
		face = img[y:h, x:w]
		return status, face

@app.get("/")
def read_root():
    return {"Welcome"}

@app.post("/uploadDocument/{side}")
async def upload_docs(side:str, file:  UploadFile = File(...)):
    info = None
    content = await file.read()
    image = np.array(bytearray(content), dtype=np.uint8)
    image = cv2.imdecode(image, 1)

    if side == 'front':
        info = get_info_citizenship(image)
    elif side == 'back':
        info = get_info_back(image)
    return {
        "info" : info
    }

@app.post("/faceSimilarity/")
async def compare_face(file1:  UploadFile = File(...), file2:  UploadFile = File(...)):
    status = False
    similarity = None
    content1 = await file1.read()
    content2 = await file2.read() 
    image1 = np.array(bytearray(content1), dtype=np.uint8)
    image1 = cv2.imdecode(image1, 1)
    image2 = np.array(bytearray(content2), dtype=np.uint8)
    image2 = cv2.imdecode(image2, 1)

    status0, face0 = get_face(image1)
    status1, face1 = get_face(image2)

    if status0 and status1:
        status = True
        similarity = find_face_similarity.matching_prediction(face0, face1)
    else:
        # if status0 == False and status1 == False:
        #     similarity = {"image_name": [file_names[0], file_names[1]], "reason": [face0, face1]}
        # elif status0 == False:
        #     similarity = {"image_name": [file_names[0]], "reason": [face0]}
        # elif status1 == False:
        #     similarity = {"image_name": [file_names[1]], "reason": [face1]}
        if not status0 and not status1:
            return {"status": status, "similarity" : "both"} # Error on both images
        elif not status0 and status1:
            return {"status": status, "similarity" : "front"} # Error on front side
        elif status0 and not status1:
            return {"status": status, "similarity" : "back"} # Error on back side
        else:
            return {"status": status, "similarity" : None}


    return {
        "status": status,
        "similarity" : similarity
    }

@app.post("/passiveLiveness/")
async def get_liveness_score(file : UploadFile = File(...)):
    liveness = None
    content = await file.read()
    image = cv2.imdecode(np.array(bytearray(content), dtype=np.uint8), 1)
    liveness = predict_liveness.predict(image)

    return {
        "liveness" : liveness
    }

@app.post("/blinkDetection/")
async def get_blink_result(file : UploadFile = File(...)):
    # blink = None
    content = await file.read()
    image = cv2.imdecode(np.array(bytearray(content), dtype=np.uint8), 1)
    blink = blink_liveness.check_blink(image)
    return {
        "blink" : blink
    }

@app.post("/headPose/")
async def get_head_pose(file : UploadFile = File(...)):
    hpose = None
    content = await file.read()

    image = cv2.imdecode(np.array(bytearray(content), dtype=np.uint8), 1)
    hpose = headpose_liveness.get_head_pose(image)
    return {
        "headpose" : hpose
    }