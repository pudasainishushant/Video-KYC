import numpy as np
import cv2

from sklearn.metrics.pairwise import cosine_similarity

import facenet, face_models

model = face_models.LoadRecogModel()

image_size = 160

def get_embedding(cropped_face):
    """Returns embedding of face image.

    Args:
        cropped_face (ndarray): input face image

    Returns:
        ndarray: face embedding
    """
    embedding_size = model.embedding_tensor()
    emb_array = np.zeros((1, embedding_size))

    scaled = []
    scaled_reshape = []
    cropped = cropped_face

    cropped = facenet.flip(cropped, False)

    scaled = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    scaled = cv2.resize(scaled, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)
    scaled = facenet.prewhiten(scaled)
    scaled_reshape = scaled.reshape(-1, image_size, image_size, 3)

    #Call function inside the loaded model
    embedding = model.embed(scaled_reshape, emb_array)
    return embedding

def matching_prediction(img1, img2):
    """prediction for image matching

    Args:
        img1 (ndarray): first input image
        img2 (ndarray): second input image

    Returns:
        int: cosine similarity value of images
    """
    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)
    similarity = cosine_similarity(emb1, emb2)

    return round(similarity[0][0], 3)