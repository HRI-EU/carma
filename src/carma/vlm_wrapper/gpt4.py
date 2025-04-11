#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  This module provides a wrapper class for OpenAI's GPT4 models.
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

from __future__ import annotations
from typing import Optional, Union

import math

import numpy
from openai import OpenAI, NOT_GIVEN

from carma.image_tools.image_tools import read_image_as_str, image_cv_to_str


class GPT4:
    detail_modes = ["high", "low"]

    def __init__(self, detail="low", model="gpt-4o", max_tokens=300, temperature=0.0):
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        if detail not in GPT4.detail_modes:
            raise ValueError(f"Unknown detail mode '{detail}'. Known values are '{GPT4.detail_modes}'.")
        self.detail = detail
        self.response = None

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

    @staticmethod
    def get_default_question_answering_text(question):
        return f"Answer as short as possible! Here is the question: {question}"

    def get_response(self) -> dict:
        return self.response.to_dict()

    def visual_question_answering(
        self,
        image: Union[numpy.ndarray, str],
        text: str,
        detail: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> Optional[str]:
        """
        Answers a question based on an image.

        :param image: The image as OpenCV array (BGR order) or encoded as string (base64/utf-8).
        :param text: The text containing the question to be answered.
        :param detail: The detail mode of the image ["low", "high"].
        :param response_format: The requested response format like 'json_object'. If not given it defaults to 'text'.
        :return: The computed answer.
        """
        if isinstance(image, numpy.ndarray):
            image_str = image_cv_to_str(image)
            image_size = image.shape[:2]
        elif isinstance(image, str):
            image_str = image
            image_size = None
        else:
            raise TypeError(f"Cannot handle images of type {type(image)}. Expected numpy.ndarray or str.")

        if detail is None:
            detail = self.detail

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/*;base64,{image_str}", "detail": detail},
                    },
                ],
            }
        ]
        self.response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format=response_format if response_format is not None else NOT_GIVEN,
        )

        print(
            f"Image size:{image_size or 'unknown'} detail:{detail}"
            f" | Tokens prompt:{self.response.usage.prompt_tokens} completion:{self.response.usage.completion_tokens}"
        )

        if len(self.response.choices) > 0:
            return self.response.choices[0].message.content

        return None

    def batch_visual_question_answering(
        self,
        images: list[Union[numpy.ndarray, str]],
        captions: list[str],
        pre_text: Optional[str] = None,
        post_text: Optional[str] = None,
        detail: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> Optional[str]:
        """
        Answers a question based on sequence of images.

        :param images: The images as list of OpenCV arrays (BGR order) or encoded as string (base64/utf-8).
        :param captions: A list containing the caption for each image.
        :param pre_text: The text put in front of the image sequence.
        :param post_text: The text put after the image sequence.
        :param detail: The detail mode of the image ["low", "high"].
        :param response_format: The requested response format like 'json_object'. If not given it defaults to 'text'.
        :return: The computed answer.
        """
        if len(images) != len(captions):
            raise AssertionError(
                f"The number of images {len(images)} is differs from the number of captions {len(captions)}."
            )

        image_strs = []
        for image in images:
            if isinstance(image, numpy.ndarray):
                image_strs.append(image_cv_to_str(image))
            elif isinstance(image, str):
                image_strs.append(image)
            else:
                raise TypeError(f"Cannot handle images of type {type(image)}. Expected numpy.ndarray or str.")

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

        messages = [{"role": "user", "content": content}]
        self.response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": response_format} if response_format is not None else NOT_GIVEN,
        )

        print(f"Tokens prompt:{self.response.usage.prompt_tokens} completion:{self.response.usage.completion_tokens}")

        if len(self.response.choices) > 0:
            return self.response.choices[0].message.content

        return None


def main():
    question = "What is it?"
    image_path = "data/french_press.jpg"
    image_str = read_image_as_str(image_path)
    gpt4 = GPT4()
    response = gpt4.visual_question_answering(image=image_str, text=question)
    print(response)


if __name__ == "__main__":
    main()
