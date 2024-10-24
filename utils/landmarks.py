import cv2
import numpy as np
import mediapipe as mp


class FaceLandmarks:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=4,
            min_detection_confidence=0.2
            )

    def get_facial_landmarks(self, frame):
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)

        landmarks = []
        for face_no, face_landmarks in enumerate(result.multi_face_landmarks):
            lm = []
            for i in range(0, 468):
                pt1 = face_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                lm.append([x, y])

            landmarks.append(np.array(lm, np.int32))

        return landmarks
