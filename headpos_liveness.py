
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

from blink_liveness import check_blink
from passive_liveness.face_detect import liveness_detector

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

head_pos_list = []
turned_left = False
turned_right = False
headpos_verification_completed = False
blinked = False
passive_liveness_completed = False


def get_head_pos(image):
    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    
    global head_pos_list, turned_left, turned_right, headpos_verification_completed, blinked, passive_liveness_completed

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image_copy = image.copy()
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []


    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360

            # print(y)

            # See where the user's head tilting
            if y < -17:
                pos = "Looking Left"
                turned_left = True
            elif y > 17:
                pos = "Looking Right"
                turned_right = True
            elif x < -17:
                pos = "Looking Down"
            else:
                pos = "Forward"

#            head_pos_list.append(pos)

#            print(turned_left, turned_right)

            if turned_left and turned_right:
                text = 'Blink Few Times'
                try:
                    image, status = check_blink(image)
                except:
                    image = image
                    status = False
                if status == True or blinked:
                    blinked = True
 #                   print(blinked, status)
                    text = 'Taking Picture for Passive Liveness'
                    passive_liveness_result = liveness_detector(image_copy)
#                    print(passive_liveness_result)
                    if passive_liveness_result:
                        text = 'Liveness Verification Completed'
                        passive_liveness_completed = True
                headpos_verification_completed = True
            elif not turned_left:
                text = 'Turn Left'
            elif not turned_right:
                text = 'Turn Right'
            else:
                text = 'Error'

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
            
            cv2.line(image, p1, p2, (255, 0, 0), 2)

            # Add the text on the image
            cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    try:
        return image, text
    except:
        return image, None



if __name__ == '__main__':
    image_path = r"C:\Users\aarya\Pictures\Camera Roll\WIN_20211125_16_44_50_Pro.jpg"
    image = cv2.imread(image_path)
    image, pos = get_head_pos(image)
    print(pos)







