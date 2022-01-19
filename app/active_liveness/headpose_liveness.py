import time
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance

mp_face_detection = mp.solutions.face_detection

def detect_face_landmark(image,min_detection_confidence=0.7):
    with mp_face_detection.FaceDetection(min_detection_confidence) as face_detection:
        h,w = image.shape[:2]
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        faces = []
        keypoints = []
        if results.detections:
            
            for detection in results.detections:

                # print('Nose tip:')
                # print(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE))
                # for m in mp_face_detection.FaceKeyPoint:
                #     print(m)
                
                face_box = detection.location_data.relative_bounding_box
                kp = list(detection.location_data.relative_keypoints)
                faces.append([  face_box.xmin * w, 
                                face_box.ymin * h, 
                                face_box.xmin * w + face_box.width * w, 
                                face_box.ymin * h + face_box.height * h])
                for i in range(6):
                    keypoints.append([int(kp[i].x * w), int(kp[i].y * h)])

        faces = np.array(faces).astype("int")

        # else:
        #     faces = None
        #     keypoints = None

        return keypoints

def get_head_pose(image):
    hpose = None

    keypoints = detect_face_landmark(image)
    if len(keypoints) != 0:
        rEye =  tuple(keypoints[0])
        lEye = tuple(keypoints[1])

        nose = np.array(keypoints[2])
        mouth = np.array(keypoints[3])
        mid = (nose + mouth)/2

        if (lEye[0]!= rEye[0]):
            slope = (lEye[1]-rEye[1])/(lEye[0]-rEye[0])
            y_incpt= rEye[1]-(slope*rEye[0])

            y = slope*mid[0] + y_incpt
            k1 = distance.euclidean(rEye, (mid[0],int(y)))
            k2 = distance.euclidean((mid[0],int(y)), lEye)

            k3 = distance.euclidean((mid[0], nose[1]), (mid[0], mouth[1]))
            k4 = distance.euclidean((mid[0],nose[1]), (mid[0],int(y)))

            Rratio = 0 if k1 == 0 else k2/k1
            Lratio = 0 if k2 == 0 else k1/k2
            print(Rratio, Lratio)
            if Rratio <= 0.5:
                hpose = "right"
            elif Lratio <= 0.5:
                hpose = "left"
            else:
                hpose = "center"
    return hpose


if __name__ == '__main__':
    # image_path = "../test/IMG_20211111_163859.jpg"
    # image = cv2.imread(image_path)
    # pose = get_head_pose(image)
    # print(pose)
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        hpose = get_head_pose(frame)
        frame = cv2.putText(frame, hpose, (20,20), 1,1,(255,0,0),2)
        cv2.imshow("result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    