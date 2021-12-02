import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase,ClientSettings
import cv2
import av
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import datetime

from methods import *
from methods_ocr import get_accurate_info
from headpos_liveness import get_head_pos, verification_completed

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
		# ctx = webrtc_streamer(ClientSettings(
        #     rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        #     media_stream_constraints={"video": True, "audio": False},
        # ),key='example', video_processor_factory=VideoProcessor)

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
			st.write('Please click the START button to start the VKYC.')

	else:
		if st.button('Take Picture'):
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
		pass
	with st.form('User Inputs'):
		
		full_name = st.text_input('Full Name', max_chars=30)
		citizenship_number = st.text_input('Citizenship number (without symbols)', max_chars=15)
		temporary_address = st.text_input('Temporary Address', max_chars=30)
		permanent_address = st.text_input('Permanent Address', max_chars=30)
		dob = st.date_input('Enter your DOB', min_value=datetime.date(1900,1,1))
		phone_no = st.text_input('Phone Number', max_chars=15)
		email_address = st.text_input('Email Address', max_chars=30)
		gender = st.selectbox('Gender', options=['Male', 'Female', 'Other'])
		st.form_submit_button()
		
		if full_name:
			st.write('Processing your documents')
			accurate_info = get_accurate_info(citizenship_front)
			print(accurate_info)

			if citizenship_number == str(accurate_info):
				st.subheader('OCR Verification Successful')
				st.subheader('Head to Next Page: Facematch Verification')
			else:
				st.subheader('OCR Verification Failed, Enter Correct Information')



elif page=='Liveness Verification':
	# ctx = webrtc_streamer(key='example', video_processor_factory=VideoProcessor)
	ctx = webrtc_streamer(
		client_settings=ClientSettings(
			rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
			media_stream_constraints={"video": True, "audio": False},
		),
		video_processor_factory=VideoProcessor,
		key="example",
	)

	if verification_completed:
		st.subheader('Liveness Verification Completed')
		st.title('All Verification Steps Completed, You Are A Verified User Now')




















#if ctx.video_transformer:
#	ctx.video_transformer.threshold1 = st.slider('Threshold1', 0,1000, 100)
#	ctx.video_transformer.threshold2 = st.slider('Threshold2', 0, 1000, 200)


