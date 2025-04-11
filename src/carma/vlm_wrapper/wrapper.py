#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  This is a wrapper around different VLM models.
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

from typing import Optional


class VLMWrapper:
    @classmethod
    def get_model(cls, model: str, detail="low", max_tokens=300, cache_dir: Optional[str] = None):
        if model == "blip2":
            from .blip2 import Blip2

            return Blip2(cache_dir=cache_dir)

        if model == "gpt4":
            from .gpt4 import GPT4

            return GPT4(detail=detail)

        if model == "gpt-4o-mini":
            from .gpt4 import GPT4

            return GPT4(detail=detail, model=model, max_tokens=max_tokens)

        if model == "llama90b":
            from .llama import Llama

            return Llama(detail=detail, model=model, max_tokens=max_tokens)

        if model == "pixtral12b":
            from .pixtral import Pixtral

            return Pixtral(detail=detail, model=model, max_tokens=max_tokens)

        if model == "dummy":
            from .dummy import Dummy

            return Dummy(detail=detail)

        raise AssertionError(f"Unknown model '{model}'. Known ones are 'blip2' and 'gpt4'.")
