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
import glob
import re
import json
import pickle
import cv2
import time
from carma.instance_detector.instance_detector import InstanceDetector
from carma.image_tools.image_tools import show_image_cv, read_image_as_cv, stitch_images, save_image_as_cv, scale_image_cv_max_size

class Baselines:
    def __init__(self, experiment_folder, model="gpt4", show_results=False, iterations=1):
        images_folder = os.path.join(experiment_folder, "images")
        object_images_folder = os.path.join(experiment_folder, "object_images")
        self.iterations = iterations
        self.image_files, self.person_ids = self.list_files_and_ids(images_folder)  
        self.experiment_folder = experiment_folder
        self.model = model
        self.results_path = os.path.join(self.experiment_folder, "runs", self.model)
        for iteration in range(iterations):
            if not os.path.exists(self.results_path + f"-{iteration}"):
                os.makedirs(self.results_path + f"-{iteration}")
                print(f"Created folder: {self.results_path}")
            else:
                json_files = glob.glob(os.path.join(self.results_path + f"-{iteration}", "*.json"))
                for file_path in json_files:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")     
        self.object_labels = self.create_object_labels()                     
        self.chunk_size = 4
        self.show_results = show_results
        self.max_size = 512
        self.instance_detector = InstanceDetector(vlm_model=model)
        self.previous_action_patterns = {}
        person_ids_string, object_labels_string = self.scene_label_mapping()
        self.instance_detector.pre_text = (
            "You are given a sequence of images. Image labels are 'image_y', where 'y' is the frame number. "
            "Your task:\n"
            "- Look only at the final image to detect the action of a person and the object it interacts with.\n"
            "- Double check if the actor really touches the object with his hands and interacts with it. "
            "If not, always use the label 'idle', never None or 'null'. Exclusively use following atomic actions: grasp, handover, place_down, "
            f"hold or pour. Following objects can appear in the images {object_labels_string} and persons {person_ids_string}\n"
            "Important:\n"
            "1. Focus on each single action-person relation sperataley.\n"
            "2. You must use one of the provided object labels if the person is interacting with it.\n"
            "3. If you identify the robot_hand of the robot in the image, verify if the robot is interacting with the "
            "human. If yes, set the item {'robot_interaction': true} if not, set it to {'robot_interaction':false}.\n"
            "4. You must include the spatial relation to a second involved object if both objects are placed in or on "
            "each other, like {person_id: {'object': 'object_1', 'action': 'place_down', 'on': 'object_3'}. The objects must be "
            "obviously in contact. If you are not sure about an object label, always use an empty string '', never None or 'null'.\n")
        self.instance_detector.post_text = (
            "Return a JSON a dict describing the action of the human actor like {67bc6c24b50c035c485bbf56: {'object': 'object_2', "
            "'action': 'hold', 'robot_interaction': False}}. Only return a single dictionary for the complete image sequence. Do never use lists."
        )
        print(self.instance_detector.pre_text)

    def create_sequence_captions(self, sequence_images):
        sequence_captions = []
        for image_ix in range(len(sequence_images)):
            image_label = f"image_{image_ix}"
            sequence_captions.append(image_label)
        return sequence_captions

    def scene_label_mapping(self):
        if "scene_009_PsortO" in self.experiment_folder:
            person_labels = ["the person in the centre"]
            object_labels_string = "object_1: bottle, object_2: dark cup, object_3: bright cup"
        elif "scene_020_sf2P" in self.experiment_folder:
            person_labels = ["the person in the centre", "the person to the left"]
            object_labels_string = "object_1: green apple, object_2: red apple, object_3: banana, object_4: black bowl, object_5: orange, object_6: blue plate"
        elif "scene_021_sf2P" in self.experiment_folder:
            person_labels = ["the person to the left", "the person in the centre"]
            object_labels_string = "object_1: green apple, object_2: red apple, object_3: banana, object_4: black bowl, object_5: orange, object_6: blue plate"
        elif "scene_022_sf2P" in self.experiment_folder:
            person_labels = ["the person to the left", "the person in the centre"]
            object_labels_string = "object_1: green apple, object_2: red apple, object_3: banana, object_4: black bowl, object_5: orange, object_6: blue plate"            
        elif "scene_026_sf1P1R" in self.experiment_folder:
            person_labels = ["the person in the centre"]
            object_labels_string = "object_1: green apple, object_2: red apple, object_3: banana, object_4: black bowl, object_5: orange, object_6: red plate, object_7: blue plate" 
        elif "scene_027_sf1P1R" in self.experiment_folder:
            person_labels = ["the person in the centre"]
            object_labels_string = "object_1: green apple, object_2: red apple, object_3: banana, object_4: black bowl, object_5: orange, object_6: red plate, object_7: blue plate" 
        elif "scene_029_sf2P1R" in self.experiment_folder:
            person_labels = ["the person to the left", "the person in the centre"]
            object_labels_string = "object_1: green apple, object_2: red apple, object_3: banana, object_4: black bowl, object_5: orange, object_6: red plate, object_7: blue plate" 
        elif "scene_0290_sf2P1R" in self.experiment_folder:
            person_labels = ["the person to the left", "the person in the centre"]
            object_labels_string = "object_1: green apple, object_2: red apple, object_3: banana, object_4: black bowl, object_5: orange, object_6: red plate, object_7: blue plate"             
        elif "scene_030_po2P" in self.experiment_folder:
            person_labels = ["the person to the left", "the person in the centre"]
            object_labels_string = "object_1: bottle, object_2: dark cup, object_3: bright cup"
        elif "scene_032_po2P" in self.experiment_folder:
            person_labels = ["the person to the left", "the person in the centre"]
            object_labels_string = "object_1: bottle, object_2: dark cup, object_3: bright cup"
        elif "scene_033_po1P1R" in self.experiment_folder:
            person_labels = ["the person in the centre"]
            object_labels_string = "object_1: bottle, object_2: dark cup, object_3: bright cup"
        elif "scene_034_po1P1R" in self.experiment_folder:
            person_labels = ["the person in the centre"]
            object_labels_string = "object_1: bottle, object_2: dark cup, object_3: bright cup"
        elif "scene_041_ha2P" in self.experiment_folder:
            person_labels = ["the person to the left", "the person in the centre"]
            object_labels_string = "object_1: bottle, object_2: dark cup, object_3: bright cup"
        elif "scene_042_ha2P" in self.experiment_folder:
            person_labels = ["the person to the left", "the person in the centre"]
            object_labels_string = "object_1: bottle, object_2: dark cup, object_3: bright cup"            
        elif "scene_043_ha1P1R" in self.experiment_folder:
            person_labels = ["the person in the centre"]
            object_labels_string = "object_1: bottle, object_2: dark cup, object_3: bright cup"
        elif "scene_044_ha1P1R" in self.experiment_folder:
            person_labels = ["the person in the centre"]
            object_labels_string = "object_1: bottle, object_2: dark cup, object_3: bright cup"
            
        person_ids_string = ", ".join(f"{pid}: {label}" for pid, label in zip(self.person_ids, person_labels))
        return person_ids_string, object_labels_string

    def create_object_labels(self):
        object_labels=[]
        return object_labels

    def write_result(self, results_path, image_filename, action_patterns):
        for person_id, action_pattern in action_patterns.items():
            results_file = f"{image_filename[:-4]}_id_{person_id}.json" 
            if not action_pattern:
                continue
            if "action" in action_pattern and action_pattern["action"] == "idle":
                print(f"{action_pattern} contains 'idle', skipping ...")
                continue
            if person_id in self.previous_action_patterns and self.previous_action_patterns[person_id] == action_pattern:
                print(f"{action_pattern} already appeared before, skipping ...")
                continue            
            self.previous_action_patterns.update({person_id: action_pattern})
            results_filename = os.path.join(results_path, results_file)  
            print(f"{action_pattern} detected, writing to {results_filename}") 
            with open(results_filename, "w") as file:
                json.dump([action_pattern], file, indent=4)


    def process(self):
        start_iterations_at = 0
        for iteration in range(start_iterations_at, self.iterations):
            processing_time = time.time()
            for i in range(0, len(self.image_files), self.chunk_size):
                image_files = self.image_files[i:i + self.chunk_size]
                sequence_images = []
                for image_filename in image_files:
                    image = read_image_as_cv(os.path.join(self.experiment_folder, "images", image_filename))
                    sequence_images.append(scale_image_cv_max_size(image, 512))
                sequence_captions = self.create_sequence_captions(sequence_images)                
                action_patterns = self.create_action_patterns(sequence_images, sequence_captions, self.object_labels)
                results_path = os.path.join(self.results_path + f"-{iteration}")  
                self.write_result(results_path, image_files[-1], action_patterns)
                if self.show_results:
                    stitched_images = stitch_images(images=sequence_images, caption_text=sequence_captions, font_size=0.5)
                    show_image_cv(stitched_images, wait_key=0)
            processing_time = time.time() - processing_time
            with open(os.path.join(results_path, 'processing_time.json'), 'w') as f:
                json.dump({"processing_time": processing_time, "images": len(self.image_files)}, f)
            

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
        # do 5 retries if reponse fails
        retries = 5
        for i in range(retries):
            try:
                response = self.instance_detector.identify_instances(images=sequence_images, image_captions=sequence_captions, response_format="json_object")
                break
            except:
                wait = 2 ** i
                print(f"Retry {i+1}/{retries} after {wait}s due to response error")
                time.sleep(wait)
                response = "{}"
        action_patterns = json.loads(response)
        return action_patterns


if __name__ == "__main__":
    runs = ["scene_009_PsortO", "scene_020_sf2P", "scene_021_sf2P", "scene_022_sf2P", "scene_026_sf1P1R", "scene_027_sf1P1R", "scene_029_sf2P1R", "scene_0290_sf2P1R",
            "scene_030_po2P", "scene_032_po2P", "scene_033_po1P1R", "scene_034_po1P1R", "scene_041_ha2P", "scene_042_ha2P", "scene_043_ha1P1R", "scene_044_ha1P1R"]
    runs = ["scene_034_po1P1R"]
    models = ["gpt-4o", "gpt-5", "gemini-2.5-flash"]
    iterations = 1
    for run in runs:
        experiment_folder = f"data/{run}"
        baselines = Baselines(experiment_folder=experiment_folder, model="gemini-2.5-flash", iterations=iterations)
        baselines.process()