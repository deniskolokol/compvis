import os
import sys
import argparse

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Take care of local modules.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils.landmarks import FaceLandmarks
from utils.resize import resize_aspect_ratio
from utils.fileops import is_video


def blur_faces(img):
    img_copy = img.copy()
    height, width, _ = img.shape

    # Face landmarks detection.
    fl = FaceLandmarks()
    try:
        landmarks = fl.get_facial_landmarks(img)
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


def video_realtime(capture, **kwargs):
    cap = cv2.VideoCapture(capture)
    cv2_major_ver = int(cv2.__version__.split('.')[0])
    if cv2_major_ver < 3:
        fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
    else:
        fps = int(cap.get(cv2.CAP_PROP_FPS))

    caption = f"Cam {capture}" if isinstance(capture, int) else capture 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # TODO: adjust to screen size (only if bigger)
        frame = resize_aspect_ratio(frame, width=1000)
        if int(kwargs.get('flip', 0)):
            frame = cv2.flip(frame, 1)
        frame = blur_faces(frame)
        cv2.imshow(caption, frame)

        key = cv2.waitKey(fps)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def image_display(img, **kwargs):
    img = cv2.imread(img)
    if int(kwargs.get('flip', 0)):
        img = cv2.flip(img, 1)

    result = blur_faces(img)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    plt.imshow(result)
    plt.show()


def main(**kwargs):
    capture = kwargs.pop("video_input", 0)
    try:
        capture = int(capture)
    except ValueError:
        pass
    else:
        return video_realtime(capture, **kwargs)
    
    if is_video(capture):
        return video_realtime(capture, **kwargs)
    else:
        return image_display(capture, **kwargs)


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
