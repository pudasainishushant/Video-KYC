import cv2
import mediapipe as mp
import time
from scipy.spatial import distance as dist

mp_facemesh = mp.solutions.face_mesh

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

def check_blink(frame):
    blink = False
    h,w,_ = frame.shape
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ear_threshold = 0.2

    with mp_facemesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        results = face_mesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLm in results.multi_face_landmarks:
                ear1 = eye_aspect_ratio([(faceLm.landmark[33].x*w, faceLm.landmark[33].y*h), (faceLm.landmark[160].x*w, faceLm.landmark[160].y*h),
                    (faceLm.landmark[158].x*w, faceLm.landmark[158].y*h), (faceLm.landmark[133].x*w, faceLm.landmark[133].y*h),
                    (faceLm.landmark[153].x*w, faceLm.landmark[153].y*h), (faceLm.landmark[144].x*w, faceLm.landmark[144].y*h)])

                ear2 = eye_aspect_ratio([(faceLm.landmark[362].x*w, faceLm.landmark[362].y*h), (faceLm.landmark[385].x*w, faceLm.landmark[385].y*h),
                    (faceLm.landmark[387].x*w, faceLm.landmark[387].y*h), (faceLm.landmark[263].x*w, faceLm.landmark[263].y*h),
                    (faceLm.landmark[373].x*w, faceLm.landmark[373].y*h), (faceLm.landmark[380].x*w, faceLm.landmark[380].y*h)])
                if ear1 <= ear_threshold or ear2 <= ear_threshold:
                    blink = True
    
    return blink

if __name__ == '__main__':
    image_path = "/home/jitfx516/Documents/images.jpeg"
    image = cv2.imread(image_path)
    print(check_blink(image))
