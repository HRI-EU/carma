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
