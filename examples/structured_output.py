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
import requests
from pydantic import BaseModel, Field

from carma.vlm_wrapper.gpt4 import GPT4
from carma.image_tools.image_tools import image_file_to_str


IMAGE_CAPTIONING = """\
Role:
You are proficient in tracking agents, objects, and actions across scene descriptions.
Your job is to add labels for agents, objects, and actions to an existing caption for image sequences.

Task:
Keep the original description of the last image and add unique labels to all agents, \
objects, and actions mentioned using square brackets. \
Also include the type, i.e., "Agent", "Object" or "Action" in the label after a colon. \
Return an adapted caption for the last image.

If you are provided with two images reuse the labels from the first image whenever applicable \
for the same objects, agents, and actions.

Here is an example response:
A person [person_1:Agent] is pouring [pour_1:Action] a bottle [bottle_1:Object]. Another person [person_2:Agent] \
is holding [hold_1:Action] a glass [glass_1:Object] until the glass [glass_1:Object] is filled up half. There are \
two further bottles visible [bottle_2:Object], [bottle_3:Object].
"""

PATTERN_EXTRACTION = """\
Extract structured action patterns from the following caption and image.

Caption:
{caption}
"""


class ActionPattern(BaseModel):
    action: str
    agents: list[str]
    objects: list[str] = Field(default_factory=list)


class ActionPatternList(BaseModel):
    action_patterns: list[ActionPattern]


image_file = "image.png"

example_image = "https://media.istockphoto.com/id/629421286/de/foto/bartender-mischt-einen-cocktail.jpg?s=1024x1024&w=is&k=20&c=4hxM6LMgfzy4wutJq7Y8fmfdqWf6OD3UxGSJ8B-jUVg="
response = requests.get(example_image)
if response.status_code == 200:
    with open(image_file, "wb") as file:
        file.write(response.content)


gpt4 = GPT4(model="gpt-4o-2024-08-06")
caption = gpt4.batch_visual_question_answering(
    images=[image_file_to_str(image_file)], pre_text=IMAGE_CAPTIONING
)
print(caption)
aps = gpt4.batch_visual_question_answering(
    images=[image_file_to_str(image_file)],
    pre_text=PATTERN_EXTRACTION.format(caption=caption),
    response_format=ActionPatternList,
)
print(aps)
