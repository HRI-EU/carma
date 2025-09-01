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
from google.genai import types
from google import genai
from pydantic import BaseModel
import json

from carma.image_tools.image_tools import image_cv_to_str
from carma.vlm_wrapper.base import VLM


class Gemini(VLM):
    def __init__(self, detail="low", model="gemini-2.5-flash"):
        super().__init__()
        self.client = genai.Client()
        self.model = model

    def batch_visual_question_answering(
        self,
        images: List[Union[np.ndarray, str]],
        captions: Optional[List[str]] = None,
        pre_text: Optional[str] = None,
        post_text: Optional[str] = None,
        response_format: str = None,
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


        content = []
        if pre_text:
            content.append(pre_text)


        for image_str, caption in zip(image_strs, captions):
            content.append([types.Part.from_bytes(data=image_str, mime_type='image/png',), 'Caption this image.'])
            if caption:
                content[-1] += [caption]

        if post_text:
            content.append(post_text)

        if response_format and response_format == "json_object":
            config={"response_mime_type": "application/json"}
            self.response = self.client.models.generate_content(model=self.model, contents=content, config=config)
        else:
            self.response = self.client.models.generate_content(model=self.model, contents=content)
        return self.response.text
