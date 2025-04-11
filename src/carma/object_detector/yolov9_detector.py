#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Use VLM on images from a camera and text from console.
#
#  Copyright (C)
#  Honda Research Institute Europe GmbH
#  Carl-Legien-Str. 30
#  63073 Offenbach/Main
#  Germany
#
#  UNPUBLISHED PROPRIETARY MATERIAL.
#  ALL RIGHTS RESERVED.
#
#

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
