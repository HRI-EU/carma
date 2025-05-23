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
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
import numpy as np
from scipy.spatial.distance import cosine, euclidean
import timm
from torchvision.transforms import InterpolationMode

import carma.image_tools.image_tools as image_tools

class VisualSimilarity:

    def __init__(self):
        # Determine the device (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device, "is used.")

        # --------------------------------------------------------
        # 1) ResNet Model
        # --------------------------------------------------------
        self.resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet_model.to(self.device)
        self.resnet_model.eval()

        self.resnet_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

        # --------------------------------------------------------
        # 2) DINO Model
        # --------------------------------------------------------
        self.dino_model = timm.create_model("vit_base_patch16_224.dino", pretrained=True)
        self.dino_model.eval()
        self.dino_model.to(self.device)

        self.dino_preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    # ------------------------------------------------------------
    # ResNet Embedding
    # ------------------------------------------------------------
    def get_resnet_embedding(self, image):
        """Extract a ResNet embedding from an image (CV2 -> PIL)."""
        img_t = self.resnet_preprocess(image_tools.image_cv_to_pil(image))
        img_t = img_t.unsqueeze(0).to(self.device)  # batch dimension
        with torch.no_grad():
            embedding = self.resnet_model(img_t)
        return embedding.squeeze().cpu().numpy()

    # ------------------------------------------------------------
    # DINO Embedding
    # ------------------------------------------------------------
    def get_dino_embedding(self, image):
        """Extract a DINO-based ViT embedding from the final layer."""
        pil_img = image_tools.image_cv_to_pil(image)
        img_t = self.dino_preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.dino_model(img_t)
        return embedding.squeeze().cpu().numpy()

    # ------------------------------------------------------------
    # Similarity Metrics
    # ------------------------------------------------------------
    @staticmethod
    def get_cosine_distance(embedding1, embedding2):
        """Cosine distance = 1 - cosine_similarity."""
        return 1 - cosine(embedding1, embedding2)

    @staticmethod
    def get_euclidean_distance(embedding1, embedding2):
        """Standard Euclidean distance."""
        return euclidean(embedding1, embedding2)

    @staticmethod
    def get_average_rgb_distance(image1, image2):
        """
        Returns the Euclidean distance between the average RGB values
        of two images. Lower distance -> more similar average color.
        """
        # Convert to numpy arrays in RGB format
        pil1 = image_tools.image_cv_to_pil(image1)
        pil2 = image_tools.image_cv_to_pil(image2)

        arr1 = np.array(pil1)  # shape: (H, W, 3)
        arr2 = np.array(pil2)

        # Ensure 3 channels (in case of 4-channel PNG or grayscale)
        if arr1.shape[-1] == 4:
            arr1 = arr1[..., :3]
        if arr2.shape[-1] == 4:
            arr2 = arr2[..., :3]

        # Compute average color
        avg_color1 = arr1.reshape(-1, 3).mean(axis=0)
        avg_color2 = arr2.reshape(-1, 3).mean(axis=0)

        # Euclidean distance in 3D color space
        distance = np.linalg.norm(avg_color1 - avg_color2)
        return distance

    @staticmethod
    def get_average_rgb_similarity(image1, image2):
        """
        Converts the average RGB distance into a similarity score in [0..1].
        - 0 => completely different (distance near 441.67 = sqrt(3*255^2))
        - 1 => identical average color (distance = 0)
        """
        dist = VisualSimilarity.get_average_rgb_distance(image1, image2)

        # Maximum possible distance in RGB is sqrt(3*(255^2)) ~ 441.67
        max_dist = np.sqrt(3 * (255 ** 2))
        similarity = 1.0 - min(dist, max_dist) / max_dist

        return similarity
