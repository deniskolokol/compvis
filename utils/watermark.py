import os
import json
import argparse
from string import ascii_letters, digits

import cv2
import numpy as np


def rand_string(size=10):
    """Generates quazi-unique sequence from random digits and letters."""
    return ''.join(np.random.choice(list(ascii_letters+digits), (size,)))


def prepare_img(path: str):
    """
    Converts BGR to RGB

    :param path: <str> Must be absolute path
    :return: <numpy.ndarray>
    """
    img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Crude image resizing.
    Adapted from the source:
    https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv

    #TODO: use interpolation for smoother result.
    """
    # Initialize the dimensions of the image to be resized and
    # grab the image size.
    dim = None
    (h, w) = image.shape[:2]

    # If both the width and height are None, return the original.
    if width is None and height is None:
        return image

    # Calculate corresponding ratio and construct the dimensions.
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation = inter)


def process_img(filepath, img_wm):
    img_base = prepare_img(filepath)
    img_watermark = img_wm

    # Check if watermark.shape is bigger than img_base and adjust accordingly.
    # If after adjustment of width, the watermark is still bigger, adjust
    # height. 
    if img_watermark.shape[1] > img_base.shape[1]:
        img_watermark = image_resize(img_watermark, width=img_base.shape[1])
    if img_watermark.shape[0] > img_base.shape[0]:
        img_watermark = image_resize(img_watermark, height=img_base.shape[0])

    # Offsets to place watermark (in the center).
    x_offset = int((img_base.shape[1] - img_watermark.shape[1]) / 2)
    y_offset = int((img_base.shape[0] - img_watermark.shape[0]) / 2)
    rows, cols, channels = img_watermark.shape

    # Calculate Region of Interest (ROI) to place blended image at the center
    # (roi.shape must be equal to img_watermark.shape)
    roi = img_base[y_offset:(y_offset+img_watermark.shape[0]),
                   x_offset:(x_offset+img_watermark.shape[1])]

    # Prepare the mask and place it on ROI.
    img_wm_gray = cv2.cvtColor(img_watermark, cv2.COLOR_RGB2GRAY)
    mask_inv = cv2.bitwise_not(img_wm_gray)
    white_bg = np.full(img_watermark.shape, 255, dtype=np.uint8)
    bg = cv2.bitwise_or(white_bg, white_bg, mask=mask_inv)
    fg = cv2.bitwise_or(img_watermark, img_watermark, mask=mask_inv)
    roi = cv2.bitwise_or(roi, fg)

    # "Glue" original image and ROI.
    y_val = y_offset + roi.shape[0]
    x_val = x_offset + roi.shape[1]
    img_base[y_offset:y_val, x_offset:x_val] = roi

    return img_base


def process(data, wm_img):
    for item in data:
        base_img = cv2.imread(item['source'])

        processed_file = process_img(item['source'], wm_img)

        base_path = item['source']
        base_path_, base_ext = item['source'].rsplit('.', 1)
        proc_path = f"{base_path_}_{rand_string()}.{base_ext}"
        cv2.imwrite(proc_path, processed_file)

        item.update({'result': proc_path})

    return data


def main(**kwargs):
    assert os.path.exists(kwargs['source']),\
           f"Source {kwargs['source']} doest not exist!"
    assert os.path.exists(kwargs['watermark']),\
           f"Watermark {kwargs['watermark']} doest not exist!"

    watermark = prepare_img(os.path.abspath(kwargs['watermark']))
    container = []
    if os.path.isfile(kwargs['source']):
        container.append({'source': os.path.abspath(kwargs['source'])})
    else:
        for root, dirs, files in os.walk(os.path.abspath(kwargs['source'])):
            for filename in files:
                container.append({'source': os.path.abspath(filename)})

    result = process(container, watermark)
    print(json.dumps(result, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add watermark to image(s).')
    parser.add_argument("source", help="Source file or directory")
    parser.add_argument("watermark", help="JPEG file with watermark")

    #TODO:
    # - add argument for positioning (upper left, upper right, center,
    #   lower left lower right)
    # - add argument for margings
    # - add argument for tiling

    args = parser.parse_args()
    main(**args.__dict__)