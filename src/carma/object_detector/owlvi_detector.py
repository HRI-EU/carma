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
import numpy as np
import cv2
import math
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image

class OWLViTDetector:
    """
    A detector using Hugging Face's OWL-ViT model for open-vocabulary object detection.
    """

    def __init__(self, conf: float = 0.3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print(f"{torch.cuda.device_count()} GPU Cores found on {torch.cuda.get_device_name(0)}.")

        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)
        self.conf = conf

    def detect_objects(
        self,
        image: np.ndarray,
        include_class_names: list = None,
        exclude_class_names: list = None,
        threshold: float = None
    ) -> dict:
        # 1. BGR -> RGB -> PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # 2. Prepare prompts
        prompts = include_class_names or ["person", "object"]
        texts = [prompts]

        # 3. Inference
        inputs = self.processor(text=texts, images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 4. Post-process
        thr = threshold if (threshold is not None) else self.conf
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=thr
        )[0]

        # 5. Build ROI dict, clamping and filtering zero-area
        object_rois = {}
        for idx, (box, score, label_idx) in enumerate(
            zip(results["boxes"], results["scores"], results["labels"])
        ):
            x1f, y1f, x2f, y2f = box.tolist()
            # floor the start, ceil the end to ensure at least 1px if any float width
            x1 = max(0, math.floor(x1f))
            y1 = max(0, math.floor(y1f))
            x2 = math.ceil(x2f)
            y2 = math.ceil(y2f)

            # skip if still zero- or negative-area
            if x2 <= x1 or y2 <= y1:
                continue

            label_str = prompts[label_idx]
            if exclude_class_names and label_str in exclude_class_names:
                continue

            label_clean = label_str.replace(" ", "_")
            instance_name = f"{label_clean}_{idx+1}"
            object_rois[instance_name] = [x1, y1, x2, y2]

        print(object_rois)

        return object_rois
