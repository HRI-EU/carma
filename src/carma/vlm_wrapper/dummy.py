#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  This module provides a dummy VLM class for testing interfaces.
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

from typing import Optional, Union

import numpy

from data_tools.image_tools import read_image_as_str


class Dummy:
    detail_modes = ["high", "low"]

    def __init__(self, detail="low"):
        if detail not in Dummy.detail_modes:
            raise ValueError(f"Unknown detail mode '{detail}'. Known values are '{Dummy.detail_modes}'.")
        self.detail = detail
        self.response = None

    @staticmethod
    def get_default_question_answering_text(question):
        return f"Answer as short as possible! Here is the question: {question}"

    def get_response(self) -> dict:
        return self.response.to_dict()

    def visual_question_answering(self, image: Union[numpy.ndarray, str], text: str) -> Optional[str]:
        """
        Answers a question based on an image.

        :param image: The image as OpenCV array (BGR order) or encoded as string (base64/utf-8).
        :param text: The text containing the question to be answered.
        :return: The computed answer.
        """
        if isinstance(image, numpy.ndarray):
            image_size = image.shape[:2]
        elif isinstance(image, str):
            image_size = None
        else:
            raise TypeError(f"Cannot handle images of type {type(image)}. Expected numpy.ndarray or str.")

        print(f"Image size:{image_size or 'unknown'} detail:{self.detail}")
        self.response = f"You gave me the instruction '{text}'. But I'm just a dummy."
        return self.response


def main():
    question = "What is it?"
    image_path = "data/french_press.jpg"
    image_str = read_image_as_str(image_path)
    dummy = Dummy()
    response = dummy.visual_question_answering(image=image_str, text=question)
    print(response)


if __name__ == "__main__":
    main()
