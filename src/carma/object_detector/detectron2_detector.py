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

import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument"
)

class Detectron2Detector:
    """
    A detector for identifying and localizing objects in images using a Detectron2 model.
    """

    def __init__(
        self,
        config_file: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        conf: float = 0.5,
    ) -> None:
        """
        Initializes the Detectron2Detector object.

        Args:
        config_file (str): Detectron2 model config (from model_zoo).
        conf (float, optional): The confidence threshold for detection.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print(f"{torch.cuda.device_count()} GPU Cores found on {torch.cuda.get_device_name(0)}.")

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf
        self.cfg.MODEL.DEVICE = self.device
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])

    def detect_objects(self, image: np.ndarray, include_class_names: list = None, exclude_class_names: list = None) -> dict:
        """
        Detect objects in an image using Detectron2.

        Args:
        image (np.ndarray): The image in which to detect objects.

        Returns:
        dict: A dictionary of detected objects and their bounding boxes.
        """
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        pred_classes = instances.pred_classes.numpy()
        pred_boxes = instances.pred_boxes.tensor.numpy()

        object_rois = {}
        object_count = 1

        for cls, box in zip(pred_classes, pred_boxes):
            class_name = self.metadata.get("thing_classes")[cls]
            class_name = class_name.replace(" ", "_")
            if include_class_names and (class_name not in include_class_names):
                continue
            if exclude_class_names and (class_name in exclude_class_names):
                continue

            instance_name = f"{class_name}_{object_count}"
            object_rois[instance_name] = [int(coord) for coord in box]
            object_count += 1

        return object_rois
