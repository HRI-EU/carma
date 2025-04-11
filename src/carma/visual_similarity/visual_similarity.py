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

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from ultralytics import YOLO

# [DINO ADDED]
import timm
from torchvision.transforms import InterpolationMode

# [CLIP ADDED]
import clip

import carma.image_tools.image_tools as image_tools


class VisualSimilarity:

    def __init__(self):
        # Determine the device (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device, "is used.")

        # --------------------------------------------------------
        # 1) YOLO Model
        # --------------------------------------------------------
        self.yolo_model = YOLO("/hri/sit/latest/Data/YOLO/v9/yolov9s.pt")
        self.yolo_model.to(self.device)
        self.yolo_model.eval()

        # --------------------------------------------------------
        # 2) ResNet Model
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
        # 3) DINO Model (Self-Supervised ViT)
        # --------------------------------------------------------
        # Example model name from timm: "vit_base_patch16_224_dino"
        # If "vit_base_patch16_224.dino" doesn't work, try with underscores.
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

        # --------------------------------------------------------
        # 4) CLIP Model (ViT-B/32)
        # --------------------------------------------------------
        # [CLIP ADDED]
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()

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
    # YOLO Embedding
    # ------------------------------------------------------------
    def get_yolo_embedding(self, image):
        try:
            features = self.get_yolo_features(image, layer_idx=-2)
        except Exception as e:
            print(f"Error extracting YOLO features: {e}")
            features = np.zeros(640)  # fallback vector
        return features

    def get_yolo_features(self, image, layer_idx=-2):
        """Extract features from a specific layer in YOLO."""
        pil_img = image_tools.image_cv_to_pil(image).resize((640, 640))
        tensor_img = torch.tensor(np.array(pil_img)).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        with torch.no_grad():
            outputs = self.yolo_model.model(tensor_img)

        if isinstance(outputs, (list, tuple)):
            features = outputs[layer_idx]
        else:
            features = outputs
        return features.cpu().numpy().flatten()

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
    # CLIP Embedding
    # ------------------------------------------------------------
    # [CLIP ADDED]
    def get_clip_embedding(self, image):
        """
        Extract a CLIP (ViT-B/32) embedding from the image.
        """
        pil_img = image_tools.image_cv_to_pil(image)
        img_t = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.encode_image(img_t)
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
