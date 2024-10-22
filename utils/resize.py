"""Various utils."""

import cv2


def resize_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    height_, width_ = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(height_)
        dim = (int(width_ * ratio), height)
    else:
        ratio = width / float(width_)
        dim = (width, int(height_ * ratio))

    return cv2.resize(image, dim, interpolation=inter)