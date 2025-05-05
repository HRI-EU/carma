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

import unittest
import sys
import os
from carma.vlm_wrapper.wrapper import VLMWrapper
from carma.image_tools.image_tools import read_image_as_cv

class TestVLMWrapper(unittest.TestCase):
    prompt = "Do you see an apple? Only answer with yes or no."
    image = read_image_as_cv(os.path.join("data", "scene_009_PsortO", "object_images", "object_2.jpg"))
    def test_smolvlm(self):
        self.vlm_wrapper = VLMWrapper.get_model(model="smolvlm")
        response = self.vlm_wrapper.visual_question_answering(image=self.image, text=self.prompt)
        answer = "correct" if "yes" in response.lower() else "wrong"
        self.assertEqual(answer, "correct")

    def test_gpt4o(self):
        self.vlm_wrapper = VLMWrapper.get_model(model="gpt4")
        response = self.vlm_wrapper.visual_question_answering(image=self.image, text=self.prompt)
        answer = "correct" if "yes" in response.lower() else "wrong"
        self.assertEqual(answer, "correct")

if __name__ == "__main__":
    unittest.main()