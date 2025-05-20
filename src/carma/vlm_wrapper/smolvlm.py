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
from typing import Optional, Union
import torch
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from transformers import AutoProcessor, AutoModelForVision2Seq
from carma.image_tools.image_tools import (
    read_image_as_cv,
    read_image_as_str,
    image_cv_to_pil,
    image_str_to_pil,
)


class SmolVLM:
    """Light weight Vision-Language-Model-Wrapper for SmolVLM-500M."""

    detail_modes = ["low", "high"]

    def __init__(
        self,
        detail: str = "low",
        model_name: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        max_tokens: int = 200,
        cache_dir: Optional[str] = None,
    ):
        if detail not in SmolVLM.detail_modes:
            raise ValueError(f"Unknown detail mode '{detail}'. Known values are {SmolVLM.detail_modes}.")
        self.detail = detail
        self.model_name = model_name
        self.max_tokens = max_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        ).to(self.device)

        self.response: Optional[str] = None

    @staticmethod
    def get_default_question_answering_text(question: str) -> str:
        return f"Answer as short as possible! Here is the question: {question}"

    def get_response(self) -> dict:
        return {"text": self.response}

    def visual_question_answering(
        self,
        image: Union[np.ndarray, str],
        text: str,
    ) -> Optional[str]:
        """
        Answers a question based on an image.

        :param image: Numpy-array or image string
        :param text: The question raised to the model
        :return: Answer or none
        """
        # Load image as PIL
        if isinstance(image, np.ndarray):
            pil_img = image_cv_to_pil(image)
        elif isinstance(image, str):
            pil_img = image_str_to_pil(image)
        else:
            raise TypeError(f"Cannot handle image of type {type(image)}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt,
            images=[pil_img],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        out = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.response = out.split("Assistant:")[-1].strip()
        return self.response

    def batch_visual_question_answering(
        self,
        images: list[Union[np.ndarray, str]],
        captions: list[str],
        pre_text: Optional[str] = None,
        post_text: Optional[str] = None,
        response_format: Optional[str] = "text",
    ) -> Optional[str]:
        """
        Answers a question based on a sequence of images.

        :param images: List of numpy arrays or image strings.
        :param captions: List of captions corresponding to each image.
        :param pre_text: Optional text placed before the images.
        :param post_text: Optional text placed after the images.
        :param response_format: The requested response format. Here the format is fixed to 'text'.
        :return: The model's answer as a string.
        """
        if len(images) != len(captions):
            raise ValueError(
                f"The number of images {len(images)} differs from the number of captions {len(captions)}."
            )

        # Convert all inputs to PIL images
        pil_images: list[Image.Image] = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil = image_cv_to_pil(img)
            elif isinstance(img, str):
                pil = image_str_to_pil(img)
            else:
                raise TypeError(f"Cannot handle image of type {type(img)}. Expected numpy.ndarray or str.")
            pil_images.append(pil)

        # Build the mixed content list
        content = []
        if pre_text:
            content.append({"type": "text", "text": pre_text})
        for pil, caption in zip(pil_images, captions):
            content.append({"type": "image"})
            if caption:
                content.append({"type": "text", "text": caption})
        if post_text:
            content.append({"type": "text", "text": post_text})

        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt,
            images=pil_images,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        out = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.response = out.split("Assistant:")[-1].strip()
        return self.response
