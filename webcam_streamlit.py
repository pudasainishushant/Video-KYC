import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings
import cv2
import av
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import datetime

import sys
import os
sys.path.insert(0, os.getcwd() + '/passive_liveness')

from methods import *
from methods_ocr import get_accurate_info, get_similarity_score
from blink_liveness import check_blink, blink_verification_completed
from headpos_liveness import get_head_pos, headpos_verification_completed, passive_liveness_completed

face_match = False


detector = MTCNN()


class VideoProcessor(VideoProcessorBase):

	def __init__(self):
		self.current_frame = np.array([])
		self.pos = 'Forward'

	def recv(self, frame):
		global page
		img = frame.to_ndarray(format='bgr24')
		self.current_frame = img.copy()

		if page == 'Liveness Verification':
			img, self.pos = get_head_pos(img)


		return av.VideoFrame.from_ndarray(img, format='bgr24')

st.title('Welcome to the AI Driven VKYC portal')

citizenship_front = st.file_uploader('Upload clear image of the front side of your citizenship')
citizenship_back = st.file_uploader('Upload clear image of the back side of your citizenship')

if citizenship_front and citizenship_back:

	citizenship_front = np.asarray(bytearray(citizenship_front.read()), dtype=np.uint8)
	citizenship_front = cv2.imdecode(citizenship_front,1)
	citizenship_back = np.asarray(bytearray(citizenship_back.read()), dtype=np.uint8)
	citizenship_back = cv2.imdecode(citizenship_back,1)

page = st.selectbox('Verification Steps', options=['OCR Verification', 'Face Match Verification', 'Liveness Verification'])

if page == 'Face Match Verification':

	if citizenship_front is None or citizenship_back is None:
		st.write('Please upload both the images correctly')
	else:
		face_doc = get_face(citizenship_front)
		face_doc_embeddings = get_embeddings(face_doc)
		st.image(cv2.cvtColor(face_doc, cv2.COLOR_BGR2RGB))

	if not face_match:
		ctx = webrtc_streamer(
			client_settings=ClientSettings(
				rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
				media_stream_constraints={"video": True, "audio": False},
			),
			video_processor_factory=VideoProcessor,
			key="example",
		)



	if not ctx.video_transformer or face_match:
		if not face_match:
			st.write('Please click the START button to start the Face Match Verification.')

	else:
		if st.button('Take Picture for Facematch Verification'):
			image = ctx.video_transformer.current_frame
			face = get_face(image)
			face_embeddings = get_embeddings(face)
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			st.image(face)

			if face_doc_embeddings.all() and face_embeddings.all():
				similarity_score = get_cosine_similarity_score(face_embeddings, face_doc_embeddings)
				if similarity_score > 0.5:
					face_match = True
					st.subheader('Facematch Verification Successful')
					st.subheader('Head to Next Page: Liveness Verification')
				else:
					st.subheader('Face Not Matched, Verification Failed!')



elif page=='OCR Verification':
	st.write('Verification Step 1 Started')
	try:
		st.image(cv2.cvtColor(citizenship_front, cv2.COLOR_BGR2RGB))
		st.image(cv2.cvtColor(citizenship_back, cv2.COLOR_BGR2RGB))
	except:
		st.stop()
	
	with st.form('User Inputs'):

		p_citizenship_number, p_name, p_gender, p_dob, p_birth_district, p_permanent_district, p_birth_ward, p_permanent_ward = get_accurate_info(citizenship_front)
		print(p_citizenship_number, p_name, p_gender, p_dob, p_birth_district, p_permanent_district, p_birth_ward, p_permanent_ward)

		if  len(str(p_citizenship_number)) < 8:
			st.write('Please upload clear images of documents and retry')
			st.form_submit_button()
			st.stop()


		full_name = st.text_input('Full Name', max_chars=30, placeholder=p_name)
		citizenship_number = st.text_input('Citizenship number (without symbols)', max_chars=15, placeholder=p_citizenship_number)
		birth_address = st.text_input('Temporary Address', max_chars=30, placeholder=p_birth_district)
		permanent_address = st.text_input('Permanent Address', max_chars=30, placeholder=p_permanent_district)
		try:
			st.write('Suggestion: '+ p_dob)
		except:
			pass
		dob = st.date_input('Enter your DOB', min_value=datetime.date(1900,1,1))
		phone_no = st.text_input('Phone Number', max_chars=15, placeholder=None)
		email_address = st.text_input('Email Address', max_chars=30, placeholder=None)
		try:
			st.write('Suggestion: '+ p_gender.capitalize())
		except:
			pass
		gender = st.selectbox('Gender', options=['Male', 'Female', 'Other'])
		st.form_submit_button()
	
		if full_name:
			
			
			if get_similarity_score(str(citizenship_number), str(p_citizenship_number)) > 0.8:
				st.subheader('OCR Verification Successful')
				st.subheader('Head to Next Page: Facematch Verification')
			else:
				st.subheader('OCR Verification Failed, Enter Correct Information')



elif page=='Liveness Verification':
	
	ctx = webrtc_streamer(
		client_settings=ClientSettings(
			rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
			media_stream_constraints={"video": True, "audio": False},
		),
		video_processor_factory=VideoProcessor,
		key="example",
	)

	if headpos_verification_completed and blink_verification_completed and passive_liveness_completed:
		st.subheader('Liveness Verification Completed')
		st.title('All Verification Steps Completed, You Are A Verified User Now')

