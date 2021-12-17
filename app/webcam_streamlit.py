from re import T
import active_liveness
import streamlit as st
from streamlit import legacy_caching
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings
import cv2
import av
import numpy as np
from ocr.back_side_parse import get_info_back, get_string_similarity
import datetime

import os, sys
from active_liveness import headpose_liveness

my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, my_absolute_dirpath + '/passive_liveness')
from passive_liveness import predict_liveness

sys.path.insert(0, my_absolute_dirpath + '/face')
from face import dnn_face_detection, find_face_similarity

face_match = False
liveness = False

def get_face(img):
	"""Returns face image

	Args:
		img (ndarray): input image

	Returns:
		ndarray: face image
	"""
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

class VideoProcessor(VideoProcessorBase):
	def __init__(self):
		self.current_frame = np.array([])
		self.pos = 'Forward'

	def recv(self, frame):
		global page
		img = frame.to_ndarray(format='bgr24')
		self.current_frame = img.copy()

		# if page == 'Liveness Verification':
		# 	img, self.pos = get_head_pos(img)
		return av.VideoFrame.from_ndarray(img, format='bgr24')

st.title('Welcome to the AI Driven VKYC portal')

citizenship_front = st.file_uploader('Upload clear image of the front side of your citizenship')
citizenship_back = st.file_uploader('Upload clear image of the back side of your citizenship')

if citizenship_front and citizenship_back:

	citizenship_front = np.asarray(bytearray(citizenship_front.read()), dtype=np.uint8)
	citizenship_front = cv2.imdecode(citizenship_front,1)
	citizenship_back = np.asarray(bytearray(citizenship_back.read()), dtype=np.uint8)
	citizenship_back = cv2.imdecode(citizenship_back,1)

page = st.selectbox('Verification Steps', options=['OCR Verification', 'FaceMatch and Liveness Verification'])

if page == 'OCR Verification':
	st.write('Verification Step 1 Started')
	try:
		stack_image = np.hstack([citizenship_front, cv2.resize(citizenship_back, \
			(citizenship_front.shape[1], citizenship_front.shape[0]))])
		st.image(cv2.cvtColor(stack_image, cv2.COLOR_BGR2RGB), width=500)
	except:
		st.stop()
	
	with st.form('User Inputs'):
		info = get_info_back(citizenship_back)
		p_citizenship_number = info['Citizenship Number']
		p_name = info['Name'].title()
		p_gender = info['Gender'] 
		p_dob = info['dob']
		p_birth_district = info['birth_place']
		p_permanent_district = info['permanent address'] 

		if len(str(p_citizenship_number)) < 8:
			st.write('Please upload clear images of documents and retry')
			st.form_submit_button()
			st.stop()

		full_name = st.text_input('Full Name', max_chars=30, placeholder=p_name)
		citizenship_number = st.text_input('Citizenship number (without symbols)', max_chars=15, placeholder=p_citizenship_number)
		birth_address = st.text_input('Birth Address', max_chars=30, placeholder=p_birth_district)
		permanent_address = st.text_input('Permanent Address', max_chars=30, placeholder=p_permanent_district)
		try:
			st.write('OCR parsed: '+ p_dob)
		except:
			pass
		dob = st.date_input('Enter your DOB', min_value=datetime.date(1900,1,1))
		phone_no = st.text_input('Phone Number', max_chars=15, placeholder=None)
		email_address = st.text_input('Email Address', max_chars=30, placeholder=None)
		try:
			st.write('OCR parsed: '+ p_gender.capitalize())
		except:
			pass
		gender = st.selectbox('Gender', options=['Male', 'Female', 'Other'])
		st.form_submit_button()
	
		if full_name:
			if get_string_similarity(str(citizenship_number), str(p_citizenship_number)) > 0.8:
				st.subheader('OCR Verification Successful')
				st.subheader('Head to Next Page: Facematch Verification')
				legacy_caching.clear_cache()
			else:
				st.subheader('OCR Verification Failed, Enter Correct Information')
				legacy_caching.clear_cache()

elif page == 'FaceMatch and Liveness Verification':
	if citizenship_front is None or citizenship_back is None:
		st.write('Please upload both the images correctly')
	else:
		status_card, face_card = get_face(citizenship_front)
		if status_card == False:
			st.subheader(f'{face_card} on uploded document!')

		else:
			st.image(cv2.cvtColor(face_card, cv2.COLOR_BGR2RGB), width = 100)

		if not face_match and not liveness:
			ctx = webrtc_streamer(
				client_settings=ClientSettings(
					rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
					media_stream_constraints={"video": True, "audio": False},
				),
				video_processor_factory=VideoProcessor,
				key="example",
			)

		if not ctx.video_transformer or face_match:
			st.write('Please click the START button to start the Face Match Verification.')

		if st.button('Take Picture for Facematch Verification'):
			selfie_image = ctx.video_transformer.current_frame
			status_selfie, selfie_face = get_face(selfie_image)
		
			if status_selfie == False:
				st.subheader(f'{selfie_face} on clicked Image!')
			else:
				stack_face = np.hstack([face_card, cv2.resize(selfie_face, \
					(face_card.shape[1], face_card.shape[0]))])						
				
				st.image(cv2.cvtColor(stack_face, cv2.COLOR_BGR2RGB), width = 200)

				face_similarity = find_face_similarity.matching_prediction(face_card, selfie_face)
				st.write(f'Face similarity : {face_similarity}')

				if face_similarity >= 0.5:
					face_match = True
					st.subheader('Facematch Verification Successful')
					st.subheader('Head to Next Page: Liveness Verification')
				else:
					st.subheader('Face Not Matched, Verification Failed!')

		if face_match and not liveness:
			head_pose_status = False
			while True:
				selfie_image = ctx.video_transformer.current_frame
				# hpose = headpose_liveness.get_head_pose(selfie_image)
				cv2.putText(selfie_image, "Turn Left", (50,50),1,1,(255,0,0),1)
				break
			

			# 	print("pose", headpose_liveness.get_head_pose(selfie_image))
	# if face_match == True:
	# 	if not liveness:
	# 		ctx = webrtc_streamer(
	# 				client_settings=ClientSettings(
	# 					rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
	# 					media_stream_constraints={"video": True, "audio": False},
	# 				),
	# 				video_processor_factory=VideoProcessor,
	# 				key="example",
	# 			)