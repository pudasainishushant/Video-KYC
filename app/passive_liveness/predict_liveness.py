import os, sys
import cv2
import numpy as np
import warnings

import time
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage

from src.utility import parse_model_name
warnings.filterwarnings('ignore')

sys.path.append('../')
from face import dnn_face_detection

my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
model_dir = my_absolute_dirpath + "/resources/anti_spoof_models"
model_test = AntiSpoofPredict(device_id=0)

models = ['4_0_0_80x80_MiniFASNetV1SE.pth', '2.7_80x80_MiniFASNetV2.pth']

# modelv1 = model_test._load_model(os.path.join(model_dir, models[0]))
# modelv2 = model_test._load_model(os.path.join(model_dir, models[1]))

def predict(image, model_dir = model_dir):
    """Returns prediction for liveness

    Args:
        image (ndarray): input image
        model_dir (str, optional): model directory path. Defaults to model_dir.

    Returns:
        str: prediction from result for input image
    """
    predText = None
    image_cropper = CropImage()

    face_box = dnn_face_detection.detect_face(image)
    if len(face_box) == 0:
        predText = "No Face Detected"
    
    elif len(face_box) >= 2:
        predText = "Multiple Face Detected"
    
    else:
        prediction = np.zeros((1, 3))
        test_speed = 0

        face_box = face_box[0]
        face_box = face_box.tolist()

        # sum the prediction from single model's result
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": face_box,
                "scale": True,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            start = time.time()

            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            test_speed += time.time()-start
            
        label = np.argmax(prediction)
        value = round(prediction[0][label]/2, 2)

        if label == 1 and value >=0.8:
            predText = "Real"
        else:
            predText = "Fake"
    
    return predText

if __name__ == '__main__':
    image = cv2.imread("../test/IMG_20211111_163859.jpg")
    print(predict(image))