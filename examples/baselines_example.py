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

import os
import re
import json
import pickle
import cv2
from carma.instance_detector.instance_detector import InstanceDetector
from carma.image_tools.image_tools import show_image_cv, read_image_as_cv, stitch_images, save_image_as_cv, scale_image_cv_max_size

class Baselines:
    def __init__(self, experiment_folder, model="gpt-4"):
        images_folder = os.path.join(experiment_folder, "images")
        object_images_folder = os.path.join(experiment_folder, "object_images")
        self.experiment_folder = experiment_folder
        self.object_labels = self.create_object_labels()
        self.image_files, self.ids = self.list_files_and_ids(images_folder)
        self.model = model
        self.chunk_size = 4
        self.show_results = True
        self.max_size = 512
        self.instance_detector = InstanceDetector(vlm_model=model)
        self.instance_detector.pre_text = (
            "You are given a sequence of images. Image labels are 'image_y', where 'y' is the frame number. "
            "Your task:"
            "- Look only at the final image to detect the action of a person and the object it interacts with. "
            "Take the captioning of the final image into account - if present." 
            "- Double check if the actor really touches the object with his hands and interacts with it. "
            "If not, use the label 'idle'. Exclusively use following atomic actions: grasp, handover, place_down, "
            "hold or pour. Following objects can appear in the images object_1: bottle, object_2: dark cup, object_3: bright cup, "
            "67bc6c24b50c035c485bbf56: person in the centre\n"
            "Important:"
            "1. Focus on a single action, person relation, considering on the previous action provided in the "
            "image label.\n"
            "2. If you identify the robot_hand of the robot in the image, verify if the robot is interacting with the "
            "human. If yes, set the item {'robot_interaction': true} if not, set it to {'robot_interaction':false}. "
            "3. You must include the spatial relation to a second involved object if both objects are placed in or on "
            "each other, like {person_id: {'object': 'object_1', 'action': 'put_down', 'on': 'object_3'}}. The objects must be "
            "obviously in contact.\n")
        self.instance_detector.post_text = (
            "Return a JSON a dict describing the action of the human actor like {67bc6c24b50c035c485bbf56: {'object': 'object_2', "
            "'action': 'hold', 'robot_interaction': False}}."
        )

    def create_sequence_captions(self, sequence_images):
        sequence_captions = []
        for image_ix in range(len(sequence_images)):
            image_label = f"image_{image_ix}"
            sequence_captions.append(image_label)
        return sequence_captions

    def create_object_labels(self):
        object_labels=[]
        return object_labels

    def write_result(self, image_filename, action_patterns):
        for person_id, action_pattern in action_patterns.items():
            results_file = f"{image_filename[:-4]}_id_{person_id}.json"            
            print(results_file)


    def process(self):
        for i in range(0, len(self.image_files), self.chunk_size):
            image_files = self.image_files[i:i + self.chunk_size]
            sequence_images = []
            for image_filename in image_files:
                image = read_image_as_cv(os.path.join(self.experiment_folder, "images", image_filename))
                sequence_images.append(scale_image_cv_max_size(image, 512))
            sequence_captions = self.create_sequence_captions(sequence_images)
            action_patterns = self.create_action_patterns(sequence_images, sequence_captions, self.object_labels)
            self.write_result(image_files[-1], action_patterns)
            if self.show_results:
                stitched_images = stitch_images(images=sequence_images, caption_text=sequence_captions, font_size=0.5)
                show_image_cv(stitched_images, wait_key=0)
            

    def list_files_and_ids(self, images_path):
        pattern_no_id = re.compile(r"^\d+\.\d+\.jpg$")
        pattern_id = re.compile(r"_id_([^.]+)\.jpg$")
        files_no_id = []
        unique_ids = set()
        for file_name in os.listdir(images_path):
            if pattern_no_id.match(file_name):
                files_no_id.append(file_name)
            else:
                m = pattern_id.search(file_name)
                if m:
                    unique_ids.add(m.group(1))
        files_no_id = sorted(files_no_id)
        unique_ids = sorted(list(unique_ids))
        return files_no_id, unique_ids

    def create_action_patterns(self, sequence_images, sequence_captions, object_labels=None):
        response = self.instance_detector.identify_instances(images=sequence_images, image_captions=sequence_captions, response_format="json_object")
        action_patterns = json.loads(response)
        return action_patterns


if __name__ == "__main__":
    experiment_folder = "/hri/localdisk/deigmoel/data_icra/scene_009_PsortO"
    baselines = Baselines(experiment_folder=experiment_folder)
    baselines.process()