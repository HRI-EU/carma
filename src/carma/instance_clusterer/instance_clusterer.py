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


from typing import Dict, List, Literal, Optional, Tuple, Union
import uuid
from typing import Dict, List, Literal
import numpy as np

# Import your existing VisualSimilarity class
from carma.visual_similarity.visual_similarity import VisualSimilarity


class InstanceClusterer:
    """
    A simple class to cluster instances (images) based on their embeddings
    from the VisualSimilarity class. Allows switching between multiple
    embedding methods (DINO, ResNet) and distance metrics (cosine/Euclidean).
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        embedding_method: Literal["dino", "resnet"] = "dino",
        use_cosine: bool = True
    ):
        """
        :param distance_threshold: If the closest cluster embedding is above this threshold,
                                   we create a new cluster.
        :param embedding_method: Which embedding to use from VisualSimilarity. Default = "dino".
        :param use_cosine: If True, use cosine distance; if False, use Euclidean distance.
        """
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method
        self.use_cosine = use_cosine

        # Instantiate your VisualSimilarity
        self.visual_similarity = VisualSimilarity()

        # Each cluster maps a unique cluster_id -> {"embeddings": [np.ndarray], "images": [np.ndarray]}
        self.clusters: Dict[str, Dict[str, List[np.ndarray]]] = {}

    def _get_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Internal helper to compute the desired embedding based on `embedding_method`.
        """
        if self.embedding_method == "dino":
            return self.visual_similarity.get_dino_embedding(image)
        elif self.embedding_method == "resnet":
            return self.visual_similarity.get_resnet_embedding(image)
        else:
            raise ValueError(f"Unknown embedding method: {self.embedding_method}")

    def _get_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Internal helper to compute the chosen distance (cosine or Euclidean).
        Lower = more similar.
        """
        if self.use_cosine:
            # Cosine distance = 1 - cosine_similarity
            return self.visual_similarity.get_cosine_distance(emb1, emb2)
        else:
            # Euclidean distance
            return self.visual_similarity.get_euclidean_distance(emb1, emb2)

    def calculate_iou(self, boxA, boxB):
        # Calculate the intersection over union by determining the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def reset_clusters(self):
        self.clusters = {}

    def create_clusters(self, images: List[np.ndarray], cluster_ids: List[str]) -> None:
        """
        Assigns each image to a cluster. We assume:
         - len(images) == len(cluster_ids)
         - each cluster_id may appear multiple times

        :param images: A list of images (np.ndarray).
        :param cluster_ids: A list of cluster IDs corresponding to each image.
        """
        if len(images) != len(cluster_ids):
            raise ValueError("Number of images must match the number of cluster IDs.")

        for img, cluster_id in zip(images, cluster_ids):
            self.clusters[cluster_id] = {"images": [img], "embeddings": [self._get_embedding(img)]}


    def add_instances(self, images: List[np.ndarray], cluster_ids: list = [], rois: list = [],
                      create_new_clusters: bool = True) -> List[
        Optional[str]]:
        n = len(images)
        overlap_threshold = 0.1
        similarity_threshold = 0.5
        skip_indices = set()

        # Pre-process to find overlapping and highly similar ROIs
        if len(rois) > len(images):
            for i in range(n):
                for j in range(i + 1, n):
                    if self.calculate_iou(rois[i], rois[j]) > overlap_threshold:
                        if self._get_distance(self._get_embedding(images[i]),
                                              self._get_embedding(images[j])) > similarity_threshold:
                            skip_indices.add(j)

        # Filter out skipped images and their ROIs
        filtered_images = [img for idx, img in enumerate(images) if idx not in skip_indices]

        # Compute embeddings for the remaining images
        embeddings = [self._get_embedding(img) for img in filtered_images]

        best_cluster = [None] * len(filtered_images)
        best_distance = [0.0] * len(filtered_images)

        for i, emb in enumerate(embeddings):
            if emb is None:
                continue
            for cluster_id, data in self.clusters.items():
                for existing_emb in data["embeddings"]:
                    if existing_emb is None:
                        continue
                    dist = self._get_distance(emb, existing_emb)
                    if dist > best_distance[i]:
                        best_distance[i] = dist
                        best_cluster[i] = cluster_id

        cluster_to_candidates = {}
        unassigned_indices = []

        # Ensure loop uses filtered_images' length
        for i in range(len(filtered_images)):
            if best_cluster[i] is not None and best_distance[i] > self.distance_threshold:
                cid = best_cluster[i]
                cluster_to_candidates.setdefault(cid, []).append(i)
            else:
                unassigned_indices.append(i)

        final_assignments = [None] * len(images)  # Keep the original list size for return
        for c_id, indices in cluster_to_candidates.items():
            if indices:
                best_img_idx = max(indices, key=lambda idx: best_distance[idx])
                # Append new detection to the cluster
                self.clusters[c_id]["embeddings"].append(embeddings[best_img_idx])
                self.clusters[c_id]["images"].append(filtered_images[best_img_idx])
                final_assignments[best_img_idx] = (c_id, best_distance[best_img_idx])

                for idx in indices:
                    if idx != best_img_idx:
                        unassigned_indices.append(idx)

        # Handle unassigned images
        if create_new_clusters:
            for idx in unassigned_indices:
                new_id = str(uuid.uuid4())
                self.clusters[new_id] = {
                    "embeddings": [embeddings[idx]],
                    "images": [filtered_images[idx]],
                }
                final_assignments[idx] = (new_id, None)
        else:
            for idx in unassigned_indices:
                final_assignments[idx] = None  # Reject by assigning None

        return final_assignments

    def filter_clusters(self, keep_last_n=None, keep_first_n=None):
        """
        Filters each cluster to keep only the last 'n' or the first 'n' elements.

        Args:
        keep_last_n (int, optional): Number of elements to keep from the end of the list in each cluster.
        keep_first_n (int, optional): Number of elements to keep from the start of the list in each cluster.

        If both keep_last_n and keep_first_n are provided, keep_last_n takes precedence.
        """

        # Check the parameters to decide the slicing strategy
        if keep_last_n is not None:
            for cluster_id in self.clusters:
                self.clusters[cluster_id]['embeddings'] = self.clusters[cluster_id]['embeddings'][-keep_last_n:]
                self.clusters[cluster_id]['images'] = self.clusters[cluster_id]['images'][-keep_last_n:]
        elif keep_first_n is not None:
            for cluster_id in self.clusters:
                self.clusters[cluster_id]['embeddings'] = self.clusters[cluster_id]['embeddings'][:keep_first_n]
                self.clusters[cluster_id]['images'] = self.clusters[cluster_id]['images'][:keep_first_n]

    def get_clusters(self, min_cluster_size=0) -> Dict[str, Dict[str, List[np.ndarray]]]:
        """
        Returns all clusters in the format:
        {
            cluster_id: {
                "embeddings": [...],
                "images": [...]
            },
            ...
        }
        """
        if min_cluster_size == 0:
            return self.clusters
        filtered_cluster = {}
        for cluster_id, cluster_data in self.clusters.items():
            if len(cluster_data["embeddings"]) > min_cluster_size:
                filtered_cluster.update({cluster_id: cluster_data})
        return filtered_cluster

    def get_cluster_images(self, cluster_id: str) -> List[np.ndarray]:
        """
        :param cluster_id: The ID of the cluster.
        :return: All images in that cluster.
        """
        return self.clusters[cluster_id]["images"] if cluster_id in self.clusters else []

    def get_cluster_embeddings(self, cluster_id: str) -> List[np.ndarray]:
        """
        :param cluster_id: The ID of the cluster.
        :return: All embeddings in that cluster.
        """
        return self.clusters[cluster_id]["embeddings"] if cluster_id in self.clusters else []


