#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2023, Honda Research Institute Europe GmbH.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     (1) Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#     (2) Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in
#     the documentation and/or other materials provided with the
#     distribution.
#
#     (3)The name of the author may not be used to
#     endorse or promote products derived from this software without
#     specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations
import base64
from io import BytesIO

import numpy
import cv2
from PIL import Image


def get_image_type(image) -> str:
    if isinstance(image, str):
        base64.b64decode(image.encode("utf-8"), validate=True)
        return "string"
    if isinstance(image, numpy.ndarray):
        return "cv2"
    if isinstance(image, Image.Image):
        return "pillow"
    return "unknown"


def convert_image(image, target_type: str):
    current_type = get_image_type(image)

    if current_type == "unknown":
        raise ValueError("Unsupported input image type.")
    if current_type == target_type:
        return image
    if current_type == "string":
        image_cv = image_str_to_cv(image)
    elif current_type == "cv2":
        image_cv = image
    elif current_type == "pillow":
        image_cv = image_pil_to_cv
    else:
        raise ValueError("Unsupported input image type.")

    # Step 2: Convert the PIL image to the target format
    if target_type == "string":
        return image_cv_to_str(image_cv)
    elif target_type == "cv2":
        return image_cv
    elif target_type == "pillow":
        return image_cv_to_pil(image_cv)

    else:
        raise ValueError(f"Unsupported target image type: {target_type}")


def image_pil_to_cv(image_pil: Image.Image) -> numpy.ndarray:
    image_np = numpy.array(image_pil)
    if image_np.ndim == 3:  # Check if it's a color image
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        return image_np


def image_str_to_cv(image_str: str) -> numpy.ndarray:
    image_bytes = base64.b64decode(image_str)
    np_array = numpy.frombuffer(image_bytes, dtype=numpy.uint8)
    image_cv = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image_cv


def image_file_to_str(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return image_bytes_to_str(image_file.read())


def image_bytes_to_str(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def image_cv_to_pil(image_cv: numpy.ndarray) -> Image.Image:
    if not isinstance(image_cv, numpy.ndarray):
        raise ValueError("Input must be a NumPy array (OpenCV image).")
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


def image_cv_to_str(image_cv: numpy.ndarray, extension: str = ".jpg") -> str:
    return image_bytes_to_str(cv2.imencode(extension, image_cv)[1].tobytes())


def read_image_as_str(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return image_bytes_to_str(image_file.read())


def read_image_as_cv(image_path: str) -> numpy.ndarray:
    return cv2.imread(image_path)


def save_image_as_cv(image: numpy.ndarray, image_path: str) -> None:
    if not isinstance(image, numpy.ndarray):
        raise ValueError("The image must be a NumPy array.")
    cv2.imwrite(image_path, image)


def show_image_cv(image: numpy.ndarray, wait_key: int = 1, destroy_all_windows: bool = True,
                  window_name: str = "Image") -> None:
    if not isinstance(image, numpy.ndarray):
        raise ValueError("The image must be a NumPy array.")
    cv2.imshow(window_name, image)
    if wait_key > -1:
        key = cv2.waitKey(wait_key) & 0xFF
        if key == ord('q'):
            exit()
    if destroy_all_windows:
        cv2.destroyAllWindows()



def image_str_to_pil(image_str: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(image_str)))


def image_cv_to_pil(image_cv: numpy.ndarray) -> Image.Image:
    return Image.fromarray(image_cv[:, :, ::-1])


def scale_image_cv(image_cv: numpy.ndarray, scale: float) -> numpy.ndarray:
    return cv2.resize(image_cv, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def resize_image_cv(image_cv: numpy.ndarray, width: int, height: int) -> numpy.ndarray:
    dsize = (width, height)
    return cv2.resize(image_cv, dsize=dsize)


def scale_image_cv_max_size(image_cv: numpy.ndarray, size: int) -> numpy.ndarray:
    height, width = image_cv.shape[:2]
    if size < height >= width:
        width, height = round(size / height * width), size
    elif size < width > height:
        width, height = size, round(size / width * height)
    else:
        return image_cv

    return cv2.resize(image_cv, (width, height), interpolation=cv2.INTER_AREA)


def scale_image_cv_to_fit_size(image_cv: numpy.ndarray, want_width: int, want_height: int) -> numpy.ndarray:
    have_height, have_width = image_cv.shape[:2]

    scale_factor_x = want_width / have_width
    scale_factor_y = want_height / have_height

    if scale_factor_x < scale_factor_y:
        new_width, new_height = want_width, round(scale_factor_x * have_height)
    else:
        new_width, new_height = round(scale_factor_y * have_width), want_height

    return cv2.resize(image_cv, (new_width, new_height), interpolation=cv2.INTER_AREA)


def crop_rois(image: numpy.ndarray, rois: dict = {}):
    cropped_images = {}
    for label, [left, top, right, bottom] in rois.items():
        cropped_image = image[top:bottom, left:right]
        cropped_images.update({label: cropped_image})

    return cropped_images


def wrap_text(text, font, max_width):
    """Wrap text based on maximum width."""
    lines = []
    words = text.split(" ")
    while words:
        line = ""
        while words and cv2.getTextSize(line + words[0], font[0], font[1], font[2])[0][0] < max_width:
            line += words.pop(0) + " "
        lines.append(line)
    return lines


def draw_rois(image: numpy.ndarray, rois: dict, show_labels: bool = True) -> numpy.ndarray:
    output_image = image.copy()

    # Loop through each ROI and draw a rectangle and label
    for obj_name, bbox in rois.items():
        x_min, y_min, x_max, y_max = bbox

        # Draw a rectangle around the ROI
        cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=2)

        # Add the object name as a label
        if show_labels:
            label_position = (x_min, y_min - 10 if y_min > 10 else y_min + 10)
            cv2.putText(
                output_image,
                obj_name,
                label_position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 0, 255),
                thickness=2,
            )

    return output_image


def stitch_images(
    images: list,
    grid_size: tuple = None,
    scale: float = 0.5,
    width = None,
    font_size: float = 1,
    line_offset: int = 15,
    border_size: int = 2,
    pre_text: str = None,
    post_text: str = None,
    caption_text: list = None,
    transpose: bool = False,
) -> numpy.ndarray:
    if grid_size:
        rows, cols = grid_size
    else:
        rows = 1
        cols = len(images)

    if len(images) == 0:
        return None

    if transpose:
        images = [images[i % cols * rows + i // cols] for i in range(len(images))]

    _height, _width = images[0].shape[:2]
    if width:
        height = int(_height * width / _width)
    else:
        width = _width
        height = _height
    height = int(height * scale)
    width = int(width * scale)

    # Define font and size for text
    font = (cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
    max_text_width = cols * (width + 2 * border_size) - 20  # 20 pixels padding

    # Wrap and calculate space for pre_text and post_text
    pre_lines = wrap_text(pre_text, font, max_text_width) if pre_text else []
    post_lines = wrap_text(post_text, font, max_text_width) if post_text else []

    pre_text_height = len(pre_lines) * line_offset  # example: line_offset pixels per line of text

    # Calculate caption height (considering it may be multi-line)
    caption_wrapped = [wrap_text(ct, font, width) for ct in caption_text] if caption_text else []
    caption_heights = [
        len(caption) * line_offset for caption in caption_wrapped
    ]  # Assuming line_offset pixels height per line

    # Determine total image height dynamically based on captions
    stitched_image_height = (
        rows * (height + 2 * border_size)
        + pre_text_height
        + sum(caption_heights)
        + (len(post_lines) * line_offset)
        + 2 * line_offset
    ) + 2 * line_offset

    stitched_image = numpy.ones((stitched_image_height, cols * (width + 2 * border_size), 3), dtype=numpy.uint8) * 255

    # Offset for rows to account for pre_text
    row_offset = pre_text_height

    # Add pre_text if any
    y_position = 10  # Start drawing text 10 pixels from the top
    for line in pre_lines:
        cv2.putText(stitched_image, line.strip(), (10, y_position), font[0], font[1], (0, 0, 0), font[2])
        y_position += line_offset

    # Calculate the bottom of the last image/caption to position post_text accordingly
    last_caption_bottom = 200

    for idx, image in enumerate(images):
        if idx >= rows * cols:
            break
        row = idx // cols
        col = idx % cols
        resized_image = cv2.resize(image, (width, height))

        bordered_image = cv2.copyMakeBorder(
            resized_image,
            border_size,
            border_size,
            border_size,
            border_size,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        y_image_position = row * (height + 2 * border_size) + row_offset + row * line_offset

        stitched_image[
            y_image_position : y_image_position + (height + 2 * border_size),
            col * (width + 2 * border_size) : (col + 1) * (width + 2 * border_size),
            :,
        ] = bordered_image

        # Adding caption for each image if provided
        caption_offset = 2 * line_offset
        caption_y_position = y_image_position + height + 2 * border_size + line_offset
        if caption_wrapped and len(caption_wrapped) > idx:
            for line in caption_wrapped[idx]:
                cv2.putText(
                    stitched_image,
                    line.strip(),
                    (col * (width + 2 * border_size) + 10, caption_y_position),
                    font[0],
                    font[1],
                    (0, 0, 0),
                    font[2],
                )
                caption_y_position += line_offset
        last_caption_bottom = max(last_caption_bottom, caption_y_position)

    # Add post_text immediately after the last caption
    y_position = last_caption_bottom + caption_offset
    for line in post_lines:
        cv2.putText(stitched_image, line.strip(), (10, y_position), font[0], font[1], (0, 0, 0), font[2])
        y_position += line_offset

    return stitched_image
