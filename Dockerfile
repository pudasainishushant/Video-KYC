FROM python:3.6
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install opencv-python
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD streamlit run webcam_streamlit.py --server.port 8051
