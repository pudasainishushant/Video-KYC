import os
import numpy as np
import cv2

my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))

modelFile = my_absolute_dirpath + "/models/dnn/res10_300x300_ssd_iter_140000.caffemodel"
configFile = my_absolute_dirpath + "/models/dnn/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

def detect_face(img,blob_size = 300, threshold=0.8):
    """ Face detection

    Args:
        img (ndarray): input image
        blob_size (int, optional): blob size. Defaults to 300.
        threshold (float, optional): threshold. Defaults to 0.8.

    Returns:
        ndarray: face box indexes
    """
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img,1.0, (blob_size, blob_size), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()

    boxes = []
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > threshold:
            box = faces[0, 0, i, 3:7]* np.array([width, height, width, height])
    
            if (0 <= box[0] <= width) and (0 <= box[1] <= height) and \
                (0 <= box[2] <= width) and (0 <= box[3] <= height):
                bb = np.round(box,2)
                # bb = np.append(bb, confidence)
                boxes.append(bb)

    return np.array(boxes)