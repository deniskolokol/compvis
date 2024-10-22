import os
import sys
import argparse

import cv2
import mediapipe as mp
import numpy as np

# Take care of local modules.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.landmarks import FaceLandmarks
from utils.resize import resize_aspect_ratio


FACE_LANDMARKS = FaceLandmarks()


def get_frame_blurred_faces(img):
    img_copy = img.copy()
    height, width, _ = img.shape

    # Face landmarks detection.
    try:
        landmarks = FACE_LANDMARKS.get_facial_landmarks(img)
    except TypeError as exc:
        return img

    convexhull = cv2.convexHull(landmarks)

    # Actual blurring.
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, convexhull, 255)

    img_copy = cv2.blur(img_copy, (37, 37))
    face_extracted = cv2.bitwise_and(img_copy, img_copy, mask=mask)

    # Extract background.
    background_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(img, img, mask=background_mask)

    # Put blurred face on the background.
    # TODO: blur the border!
    return cv2.add(background, face_extracted)


def main(**kwargs):
    flip = bool(kwargs.get("flip", 1))
    video_capture = kwargs.get("video_input", 0)
    try:
        video_capture = int(video_capture)
    except ValueError:
        pass

    cap = cv2.VideoCapture(video_capture)
    cv2_major_ver = int(cv2.__version__.split('.')[0])
    if cv2_major_ver < 3:
        fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    else:
        fps = int(cap.get(cv2.CAP_PROP_FPS))

    caption = video_capture if video_capture else f"Cam {video_capture}"
    while True:
        ret, frame = cap.read()

        # TODO: adjust to screen size (only if bigger)
        frame = resize_aspect_ratio(frame, width=2000)
        frame = get_frame_blurred_faces(frame)
        cv2.imshow(caption, frame)

        key = cv2.waitKey(fps)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='display')
    parser.add_argument('--input',
                        type=str,
                        default=0,
                        metavar='VIDEO_INPUT',
                        dest='video_input',
                        help='Video capture index (webcam, external cam, etc.) or mp4 filename (default: 0, i.e. built-in camera)')
    parser.add_argument('--flip',
                        action=argparse.BooleanOptionalAction,
                        dest='flip',
                        default=False,
                        help='Flip the image horizontally for a selfie-view display (default: False)')
    args = parser.parse_args()
    main(**vars(args))
