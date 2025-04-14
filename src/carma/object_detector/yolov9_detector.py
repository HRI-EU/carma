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

import numpy as np
import torch
from ultralytics import YOLO


class YoloDetectorV9:
    """
    A detector for identifying and localizing objects in images using the YOLO model.
    """

    def __init__(
        self, yolo_model: str = "/hri/sit/latest/Data/YOLO/v9/yolov9s.pt", image_size: int = 640, conf: float = 0.5
    ) -> None:
        """
        Initializes the YoloDetectorV9 object.

        Args:
        yolo_model (str): The YOLO model specification.
        image_size (int, optional): The size to which images will be resized before detection. Defaults to 640.
        conf (float, optional): The confidence threshold for detection. Defaults to 0.5.
        """
        self.image_size = image_size
        self.conf = conf
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print(f"{torch.cuda.device_count()} GPU Cores found on {torch.cuda.get_device_name(0)}.")
        self.model = YOLO(yolo_model)

    def detect_objects(self, image: np.ndarray, include_class_names: list = None, exclude_class_names: list = None) -> dict:
        """
        Detect objects in an image using the YOLO model.

        Args:
        image (np.ndarray): The image in which to detect objects.

        Returns:
        dict: A dictionary of detected objects and their bounding boxes.
        """
        results = self.model.predict(image, conf=self.conf)
        object_rois = {}
        object_count = 1
        for result in results:
            for box in result.boxes:
                class_name = result.names[int(box.cls[0])]
                if include_class_names and (class_name not in include_class_names):
                    continue
                if exclude_class_names and (class_name in exclude_class_names):
                    continue
                instance_name = f"{result.names[int(box.cls[0])]}_{object_count}"
                object_rois.update(
                    {
                        instance_name: [
                            int(box.xyxy[0][0]),
                            int(box.xyxy[0][1]),
                            int(box.xyxy[0][2]),
                            int(box.xyxy[0][3]),
                        ]
                    }
                )
                object_count += 1
        return object_rois
