# vkyc
[![PyPI](https://img.shields.io/pypi/v/danger-python)](https://pypi.org/project/danger-python/)
![Python versions](https://img.shields.io/pypi/pyversions/danger-python)
[![Build Status](https://travis-ci.org/danger/python.svg?branch=master)](https://travis-ci.org/danger/python)

# Setup Steps
- pip install -r requirements.txt
- pip install "fastapi[all]"
- sudo apt update
- sudo apt install tesseract-ocr
- sudo apt install libtesseract-dev

# Running services
## API service
- uvicorn main:app --reload
## Streamlit webapp
- streamlit run webcam_streamlit.py