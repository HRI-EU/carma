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
#
#
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Type
import numpy as np
from pydantic import BaseModel


class VLM(ABC):
    detail_modes = ["low", "high"]

    def __init__(self) -> None:
        self.response = None

    @staticmethod
    def get_default_question_answering_text(question):
        return f"Answer as short as possible! Here is the question: {question}"

    def get_response(self) -> dict:
        return {"text": self.response}

    @abstractmethod
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
        raise NotImplementedError
