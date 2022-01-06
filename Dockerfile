FROM tiangolo/uwsgi-nginx-flask:python3.8

WORKDIR /app
ENV CGO_CFLAGS="-I/usr/lib/x86_64-linux-gnu/hdf5/serial/include"
ENV LIBRARY_PATH /usr/lib/x86_64-linux-gnu/hdf5/serial/
RUN apt-get -y update \
  && apt-get install -y \
    build-essential \
    libpcre3 \
    libpcre3-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    software-properties-common \
    ffmpeg \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/
RUN pip install --ignore-installed -r requirements.txt --no-cache-dir
RUN sudo apt update
RUN apt install uvicorn
RUN sudo apt install tesseract-ocr
RUN sudo apt install libtesseract-dev
ADD ./app /app
CMD uvicorn main:app --reload