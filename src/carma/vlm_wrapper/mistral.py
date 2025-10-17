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
from typing import List, Optional, Union, Type

import math

import numpy as np
from openai import OpenAI, NOT_GIVEN
from pydantic import BaseModel

from carma.image_tools.image_tools import image_cv_to_str
from carma.vlm_wrapper.base import VLM


class Mistral(VLM):
    def __init__(self, detail="low", model="HRI-MISTRAL", max_tokens=300, temperature=1e-9):
        super().__init__()
        self.client = OpenAI(base_url="https://hri-llm.honda-ri.de/v1", api_key="empty")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.detail = detail

    def calculate_image_tokens(self, width: int, height: int) -> int:
        """
        Compute the tokens per image.

        :param width: The width of the image.
        :param height: The height of the image.
        :return: The number of tokens.
        """
        if self.detail == "low":
            return 85

        # Ensure that longest side is not larger than 2048.
        if 2048 < height >= width:
            width, height = round(2048 / height * width), 2048
        elif 2048 < width > height:
            width, height = 2048, round(2048 / width * height)

        # Ensure that smallest side is not larger than 768.
        if 768 < height >= width:
            width, height = round(768 / height * width), 768
        elif 768 < width > height:
            width, height = 768, round(768 / width * height)

        tiles_width = math.ceil(width / 512)
        tiles_height = math.ceil(height / 512)
        return 85 + 170 * (tiles_width * tiles_height)

    def batch_visual_question_answering(
        self,
        images: List[Union[np.ndarray, str]],
        captions: Optional[List[str]] = None,
        system_text: Optional[str] = None,
        pre_text: Optional[str] = None,
        post_text: Optional[str] = None,
        detail: Optional[str] = None,
        response_format: Optional[Union[str, Type[BaseModel]]] = None,
    ) -> Optional[str]:
        """
        Answers a question based on sequence of images.

        :param images: The images as list of OpenCV arrays (BGR order) or encoded as string (base64/utf-8).
        :param captions: A list containing the caption for each image.
        :param system_text: The system message.
        :param pre_text: The text put in front of the image sequence.
        :param post_text: The text put after the image sequence.
        :param detail: The detail mode of the image ["low", "high"].
        :param response_format: The requested response format like 'json_object'. If not given it defaults to 'text'.
        :return: The computed answer.
        """
        if captions is None:
            captions = ["" for _ in range(len(images))]
        if len(images) != len(captions):
            raise AssertionError(
                f"The number of images {len(images)} is differs from the number of captions {len(captions)}."
            )

        image_strs = []
        for image in images:
            if isinstance(image, np.ndarray):
                image_strs.append(image_cv_to_str(image))
            elif isinstance(image, str):
                image_strs.append(image)
            else:
                raise TypeError(f"Cannot handle images of type {type(image)}. Expected np.ndarray or str.")

        if detail is None:
            detail = self.detail

        content = []
        if pre_text:
            content.append({"type": "text", "text": pre_text})

        for image_str, caption in zip(image_strs, captions):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_str}", "detail": detail},
                }
            )
            if caption:
                content.append({"type": "text", "text": caption})

        if post_text:
            content.append({"type": "text", "text": post_text})

        messages = [{"role": "system", "content": system_text}] if system_text else []
        messages.append({"role": "user", "content": content})
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        params["response_format"] = {"type": response_format} if response_format is not None else NOT_GIVEN
        print(params["response_format"])
        self.response = self.client.chat.completions.create(**params)

        print(f"Tokens prompt:{self.response.usage.prompt_tokens} completion:{self.response.usage.completion_tokens}")

        if len(self.response.choices) > 0:
            return self.response.choices[0].message.content

        return None
