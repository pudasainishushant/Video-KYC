import cv2
import warnings
import numpy as np

import sys
import os
sys.path.insert(0, os.getcwd() + '/passive_liveness/src')

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from util import *


warnings.filterwarnings('ignore')

# init
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
i = 0
camera = cv2.VideoCapture(0)
model_test = AntiSpoofPredict(0)
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()


def face_detector(frame):
    image_bbox = model_test.get_bbox(frame)
    img = frame.copy()
    cv2.rectangle(
        img,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        (255, 0, 0), 2)

    return img


def liveness_detector(frame):
    image_cropper = CropImage()
    model_dir = 'passive_liveness/resources/liveness_model'
    image_bbox = model_test.get_bbox(frame)
    if image_bbox[0] == 0 and image_bbox[1] == 0 and image_bbox[2] == 1 and image_bbox[3] == 1:
        return False
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))

    # label: face is true or fake
    label = np.argmax(prediction)
    # value: the score of prediction
    value = prediction[0][label]
    if label == 1 and value > 0.7:
        return True
    else:
        return False



if __name__ == '__main__':
    img = cv2.imread(r"C:\Users\aarya\Downloads\WIN_20211201_12_13_51_Pro.jpg")
    print(liveness_detector(img))