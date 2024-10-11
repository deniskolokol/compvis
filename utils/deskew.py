"""
Skew correction of text (or text-like images).
"""

import os
import argparse

import numpy as np
import cv2


def get_angle(img, **kwargs):
    # Converts the image to grayscale and flip the foreground and background
    # to ensure foreground is now 'white' and the background is 'black'.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # If conversion to grayscale is selected, avoid doing it twice - return
    # converted image along with the angle.
    if kwargs.get('grayscale', False):
        img = gray
    gray = cv2.bitwise_not(gray)

    # Threshold the image - set all foreground pixels to 255
    # and all background pixels to 0.
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Use all (x, y) coordinates of the non-zero pixes to compute a rotating
    # bounding box that contains all coordinates.
    coords = np.column_stack(np.where(thresh > 0))

    # The `cv2.minAreaRect` function returns values in the range (-90, 0);
    angle = cv2.minAreaRect(coords)[-1]
    # Corrections - see
    # https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/
    if angle in (90, -90, 0):
        # -90 and 90 is the same as 0 (this depends on the orientation of
        # the rectangle) - no rotation needed.
        angle = 0
    elif angle > 45:
        # as the rectangle rotates clockwise, and returned angle
        # trends to 90 - find difference between the angle and 90;
        angle = 90 - angle
    elif angle < -45:
        # as the rectangle rotates counter-clockwise, and the angle
        # trends to 0 - add 90 degrees to the angle;
        angle += 90
    else:
        angle = -angle

    return angle, img


def deskew_img(angle, img, output_path):
    # Rotate the image to de-skew it.
    if angle == 0:
        corrected = img
    else:
        (height, width) = img.shape[:2]
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, -angle, scale=1)
        corrected = cv2.warpAffine(
            img,
            M,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
    cv2.imwrite(output_path, corrected)


def process(data, **kwargs):
    for item in data:
        img_src = cv2.imread(item['source'])
        angle, img_to_convert = get_angle(img_src, **kwargs)
        item.update({'angle': angle})
        deskew_img(angle, img_to_convert, item['output'])
        print("source: {source} -> result: {output} (angle: {angle:.5f})"\
              .format(**item))

    return data


def get_output_path(fpath, is_file, **kwargs):
    """Figure out the path to save current file."""
    # No output selected.
    output = kwargs.get('output', None)
    if output is None:
        file_path, file_ext = fpath.rsplit('.', 1)
        return f'{file_path}_deskewed.{file_ext}'

    # Single file processing - output filename given.
    if is_file:
        return output

    # Directory processing - kwargs['output'] is a directory.
    assert os.path.exists(output), f"Output {output} doest not exist!"
    return os.path.join(output, os.path.basename(fpath))


def main(**kwargs):
    assert os.path.exists(kwargs['source']),\
           f"Source {kwargs['source']} doest not exist!"

    container = []
    if os.path.isfile(kwargs['source']):
        container.append({
            'source': os.path.abspath(kwargs['source']),
            'output': get_output_path(kwargs['source'], is_file=True, **kwargs)
            })
    else:
        for root, dirs, files in os.walk(os.path.abspath(kwargs['source'])):
            for filename in sorted(files):
                fpath = os.path.join(root, filename)
                container.append({
                    'source': os.path.join(root, filename),
                    'output': get_output_path(fpath, is_file=False, **kwargs)
                    })

    data = process(container, **kwargs)
    print(f"\nDone: {len(data)} files processed.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Skew correction of text-like images.')
    parser.add_argument('-g', '--grayscale',
                        action='store_true',
                        default=False,
                        help="convert result to grayscale (default: False)")
    parser.add_argument('-o', '--output',
                        required=False,
                        help="output file or directory (if not specified, look for files with the name XXX_deskewed.ZZZ in the same directory with the source file(s)")
    parser.add_argument('source', help="Source file or directory")
    
    args = parser.parse_args()
    main(**args.__dict__)