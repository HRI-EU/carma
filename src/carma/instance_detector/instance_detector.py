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

"""
This module defines a PersonDetector class that uses a Vision-Language Model (VLM) to identify and match persons
across different images. It utilizes the VLMWrapper to interact with a specific VLM model, processes image inputs,
and returns a JSON dictionary of identified matches with confidence scores and explanations.
"""

from __future__ import annotations
from typing import Union
import json
import numpy

from carma.vlm_wrapper.wrapper import VLMWrapper


class InstanceDetector:
    """
    A detector for identifying and matching persons in images using a VLM.

    Attributes:
        vlm_model (VLMWrapper): The VLM model used for detection.
        pre_text (str): Pre-question text to be appended before image processing.
        post_text (str): Post-question text to be appended after image processing.
    """

    def __init__(self, vlm_model: str, prompt_version: int = 1):
        """
        Initializes the PersonDetector with a specified VLM model and configuration.

        Args:
            vlm_model: The identifier for the VLM model to be used.
        """
        self.vlm_model = VLMWrapper.get_model(vlm_model)
        # Define pre_text and post_text versions as a dictionary for cleaner organization
        self.text_versions = {
            "pre_text": {
                0: (
                    "Can you find the object visible in the first image again in the second image?"
                ),
                1: (
                    "You are given a sequence of images. The first images serve as references for labeling objects, "
                    "actions, and persons. Each reference image has a caption in the form [label_x:Object], "
                    "[label_y:Action], [label_z:Person] only, or embedded in a description. The final image in the "
                    "sequence includes a request in the caption of the form [?:Action, ?:Object, ?:Person]. "
                    "\n\nYour task:\n"
                    "- Look only at the final image to detect any persons, the actions they perform, and the objects "
                    "they interact with.\n" 
                    "- Use or adapt the labels provided by the reference images (where applicable). Focus on the "
                    "person’s hands.\n"
                    "- If new labels are needed for newly observed objects/actions/persons, create them similarly "
                    "(e.g., cup_2:Object, drink_1:Action, person_2:Person).\n\n"
                    "Produce a single JSON dictionary with:\n"
                    "- 'action_patterns': A list of dictionaries describing distinct interactions, with keys like "
                    "'<label>:<Type>: <confidence_score>'.\n"
                    "- 'description': A short sentence describing recognized persons, objects, and actions.\n\n"
                    "Important:\n"
                    "1. Confidence scores are floats between 0 and 1.\n"
                    "2. Detect multiple actions/objects/persons if it makes sense.\n"
                    "3. Ignore background details not part of interactions.\n"
                    "4. Return only the JSON object, with one dictionary per interaction in 'action_patterns'.\n\n"
                    "Example:\n"
                    "{'action_patterns': [{'cup_1:Object': 0.9, 'hold_1:Action': 0.8, 'person_1:Person': 1.0}], "
                    "'description': 'Person [person_1:Person] holds [hold_1:Action] [cup_1:Object].'}"
                ),
                2: (
                    "You are given a sequence of images. The first images serve as references for objects and "
                    "persons of special interest. Each reference image has a caption in the form [label_x:Object] or "
                    "[label_z:Person]. The reference images are split into an upper part and a lower part. The upper "
                    "part shows the original reference, whereas the lower part is a best matching candidate, cropped "
                    "from the final image, using visual similarity measure. Images after the reference images are "
                    "preceding images of the final image and include a description, including labels of objects, "
                    "persons and actions. The final image in the sequence includes a request in the caption of the "
                    "form [?:Action, ?:Object, ?:Person]. "
                    "\n\nYour task:\n"
                    "1. Look only at the final image to detect any persons, the actions they perform, and the objects "
                    "they interact with.\n"
                    "2. Detect and list any other objects seen, even if not part of an interaction.\n"
                    "3. Use the labels provided by reference images and take the proposed copped image in te lower "
                    "reference image part into account. \n"
                    "4. If new labels are needed for newly observed objects, actions, or persons, create them "
                    "consistently (e.g., cup_2:Object, drink_1:Action, person_2:Person).\n\n"
                    "Produce a JSON dictionary with:\n"
                    "- 'detected_objects': A list of objects detected, each as {'label:Type': confidence}.\n"
                    "- 'action_patterns': A list of dictionaries describing interactions.\n"
                    "- 'description': A sentence describing recognized persons, objects, and actions.\n\n"
                    "Important:\n"
                    "1. Confidence scores are floats between 0 and 1.\n"
                    "2. Detect multiple actions/objects/persons if it makes sense.\n"
                    "3. Ignore unnecessary details in the background, but list visible standalone objects under "
                    "'detected_objects'.\n"
                    "4. Return only the JSON object.\n\n"
                    "Example:\n"
                    "{'detected_objects': [{'cup_1:Object': 0.9}], "
                    "'action_patterns': [{'cup_1:Object': 0.9, 'hold_1:Action': 0.8, 'person_1:Person': 1.0}], "
                    "'description': 'Person [person_1:Person] holds [hold_1:Action] [cup_1:Object].'}"
                ),
            },
            "post_text": {
                0: (
                    "Return only yes or no"
                ),
                1: (
                    "Return a JSON dictionary in this format:\n"
                    "{'action_patterns': [{'cup_1:Object': 0.9, 'hold_1:Action': 0.8, 'person_1:Person': 1.0}], "
                    "'description': 'Person [person_1:Person] holds [hold_1:Action] [cup_1:Object].'}"
                    "\n\nDo not include additional text, explanations, or commentary in your final output—only the "
                    "JSON object.\n\n"
                    "For each interaction you find, add a dictionary to 'action_patterns'. Use reference image "
                    "labels if available, for create new ones. Ensure confidence scores are between 0 and 1."
                ),
                2: (
                    "Return a JSON dictionary in the format:\n"
                    "'detected_objects': [{'label:Type': confidence}, ...], "
                    "'action_patterns': [{'label:Type': confidence, ...}], "
                    "'description': 'Short sentence describing persons, objects, and actions.'\n\n"
                    "Do not include additional text, explanations, or commentary—only the JSON object."
                ),
            },
        }

        # Use helper method to select the appropriate text
        self.pre_text = self._select_text("pre_text", prompt_version)
        self.post_text = self._select_text("post_text", prompt_version)

    def _select_text(self, text_type, version):
        """Helper method to select the appropriate text version."""
        return self.text_versions[text_type].get(version, "")

    def append_to_pre_text(self, text_to_append: str):
        self.pre_text += text_to_append

    def identify_instances(self, images: list[Union[numpy.ndarray, str]], image_captions: list[str],
                           response_format: str="json_object") -> dict[str, dict]:
        """
        Identifies and matches persons in the given sample images to the reference image.

        Args:
            images: The reference image to find persons in.
            image_captions: Optional captions for all the images; defaults to generic some generic identifiers.

        Returns:
            A dictionary with frame numbers as keys and dicts with 'confidence' and 'explanation' as values.
        """
        if len(image_captions) != len(images):
            raise AssertionError(
                f"The number of images {len(images)} is differs from the number of captions {len(image_captions)}."
            )
        response = self.vlm_model.batch_visual_question_answering(
            images=images,
            captions=image_captions,
            pre_text=self.pre_text,
            post_text=self.post_text,
            response_format=response_format
        )

        return response
