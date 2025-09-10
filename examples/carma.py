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

import json
import os
import pickle
import glob
import shutil
import time

from carma.instance_clusterer.instance_clusterer import InstanceClusterer
from carma.instance_detector.instance_detector import InstanceDetector
from carma.object_detector.owlvi_detector import OWLViTDetector
from carma.image_tools.image_tools import (
    save_image_as_cv,
    read_image_as_cv,
    show_image_cv,
    crop_rois,
    stitch_images,
    scale_image_cv_max_size,
    draw_rois,
)


def load_object_images(object_image_files):
    object_images = {}
    for object_images_file in object_image_files:
        print("loading", object_images_file)
        object_image = read_image_as_cv(object_images_file)
        object_label = os.path.basename(object_images_file[:-4])
        object_images.update({f"{object_label}": object_image})
    return object_images


def load_pickle(person_actions_file):
    # print(f"Loading: {person_actions_file}")
    with open(person_actions_file, "rb") as file:
        return pickle.load(file)


def export_results(export_file, responses):
    with open(export_file, "w") as file:
        json.dump(responses, file, indent=4)


def get_filenames(data_path, object_images_folder="object_images", person_actions_folder="person_actions"):
    object_images_path = os.path.join(data_path, object_images_folder)
    person_actions_path = os.path.join(data_path, person_actions_folder)

    object_images_files = sorted(
        [os.path.join(object_images_path, f) for f in os.listdir(object_images_path) if f.endswith(".jpg")]
    )
    person_actions_files = sorted(
        [os.path.join(person_actions_path, f) for f in os.listdir(person_actions_path) if f.endswith(".pkl")]
    )

    return object_images_files, person_actions_files


class Carma:
    def __init__(self, object_images, use_ocad_labels=True, use_ocad_trigger=False, model="gpt-4o"):
        self.similarity_threshold = 0.0
        self.frame_count = 0
        self.model = model
        self.use_ocad_labels = use_ocad_labels
        self.use_ocad_trigger = use_ocad_trigger
        self.object_images = object_images
        self.instance_clusterer = InstanceClusterer(distance_threshold=self.similarity_threshold)
        self.instance_clusterer.create_clusters(list(self.object_images.values()), list(self.object_images.keys()))
        self.instance_detector = InstanceDetector(vlm_model=model)
        self.object_detector = OWLViTDetector(conf=0.007)
        self.previous_actions = {}
        self.previous_action_patterns = {}
        self.image_buffer = {}
        self.image_quadrupels = []
        self.instance_detector.pre_text = (
            "You are given a sequence of images for objects, persons and four consecutive images that might contain object or person. " 
            "Each object image has a caption in the form 'object_x' for labeled object images, where 'x' is the "
            "ID of the object. For person images, the caption contains the person ID and the action the person probbably performs like "
            "this 'person_id: action_label'. The four consecutive images are captioned by 'image_y', where 'y' is the frame number.\n"
            "Your task:\n"
            "- Look only at the final image captioned with 'Caption this image', to detect if a person performs an action and the object "
            " it interacts with. If present, take the action provided in the person caption into account and verify if the caption is "
            "correct\n." 
            "- Double check if the person really touches the object with his hands and interacts with it. "
            "If not, use the label 'idle'. Exclusively use following atomic actions: grasp, handover, place_down, hold or pour.\n"
            "Important:\n"
            "1.  Focus on each single action-person relation sperataley.\n"
            "2. You must use one of the provided object labels if the person is interacting with it.\n"
            "3. If you identify the robot_hand of the robot in the image, verify if the robot is interacting with the human. "
            " If yes, set the item {'robot_interaction': true} if not, set it to {'robot_interaction’: false}.\n"
            "4. You must include the spatial relation to a second involved object if two objects are placed in or on each other, "
            " like {'object': 'object_1', 'action': 'place_down', 'on': 'object_3’}. The objects must be obviously in contact.\n"
        )
        self.instance_detector.post_text = (
            "Return a JSON dict describing the action of the acting persons like {67bc6c24b50c035c485bbf56: {'object': 'object_2', "
            "'action': 'hold', 'robot_interaction': False}. Only return a single dictionary for the complete image sequence. "
        )

    def add_image(self, image):
        self.image_quadrupels.append(image)
        if len(self.image_quadrupels) > 4:
            self.image_quadrupels.pop(0)
        return self.image_quadrupels

    def write_result(self, results_path, image_filename, action_patterns):
        for person_id, action_pattern in action_patterns.items():
            results_file = f"{image_filename[:-4]}_id_{person_id}.json" 
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


    def create_action_patterns(self, action_images, action_captions, object_images, object_captions):
        images = object_images + action_images
        # do 5 retries if reponse fails
        retries = 5
        for i in range(retries):
            try:
                response = self.instance_detector.identify_instances(images, image_captions=object_captions + action_captions, response_format="json_object")
                break
            except:
                wait = 2 ** i
                print(f"Retry {i+1}/{retries} after {wait}s due to response error")
                time.sleep(wait)
                response = "{}"
        action_patterns = json.loads(response)
        for actor_type, action_pattern in action_patterns.items():
            # remove if idle action
            if (actor_type == "robot_interaction") or (action_pattern is None):
                continue
            if "action" == actor_type and action_pattern == "idle":
                action_patterns = {}
            # remove if invalid object label
            if "object" == actor_type and (action_pattern not in list(self.object_images.keys())):
                action_patterns = {}
            for spatial_relation in ["in", "on"]:
                if spatial_relation == actor_type and (action_pattern not in list(self.object_images.keys())):
                    action_patterns = {}
        return action_patterns

    def process(self, image_quadrupel, person_actions, analyze=True):
        action_patterns = {}
        object_images = list(self.object_images.values())
        object_captions = list(self.object_images.keys())
        person_images = []
        person_captions = []
        image_captions = []
        for person_id, data in person_actions.items():
            person_images.append(data["image"])
            person_captions.append(f"{person_id}: {data['action']}")
        for image_idx in range(len(image_quadrupel)):
            image_caption = f"image_{image_idx}"
            if image_idx == len(image_quadrupel) - 1:
                image_caption += ": Caption this image."
            image_captions.append(image_caption)
        if len(person_captions) > 0:
            action_patterns = self.create_action_patterns(image_quadrupel, image_captions, object_images + person_images, 
                                                          object_captions + person_captions)
            stitched_image = stitch_images(images=object_images+person_images+image_quadrupel, 
                                           caption_text=object_captions+person_captions+image_captions,scale=1.0)
            show_image_cv(stitched_image, wait_key=0)
        return action_patterns
    
def get_sorted_imagefiles(images_path):
    image_files = []
    for image_file in os.listdir(images_path):
        if "_id_" in image_file:
            continue
        else:
            image_files.append(image_file)
    return sorted(image_files)



def main(run_settings, runs, base_folder, show_images, write_results, create_ground_truth, iterations):

    start_iterations_at = 0
    # ########################## MAIN LOOP ############################################
    for run in runs:
        for iteration in range(start_iterations_at, iterations):
            for run_setting in run_settings:
                data_path = os.path.join(base_folder, run)    
                images_path = os.path.join(data_path, "images")
                image_filenames = get_sorted_imagefiles(images_path=images_path)
                nb_images = len(image_filenames)
                use_ocad_labels = True if "label" in run_setting[0] else False
                use_ocad_trigger = True if "trigger" in run_setting[1] else False
                model = "gpt-4o" if run_setting[2] == "" else run_setting[2]
                run_name = f"{run_setting[1]}-{run_setting[0]}-{model}-{iteration}"
                export_folder = f"{data_path}/runs/{run_name}"
                if not os.path.exists(export_folder):
                    os.makedirs(export_folder)
                elif write_results:
                    patterns = ['*.json', '*.jpg']
                    files = []
                    for pattern in patterns:
                        files.extend(glob.glob(os.path.join(export_folder, pattern)))
                    for file in files:
                        print(f"removing {file}")
                        os.remove(file)

                object_image_files, person_action_files = get_filenames(data_path=data_path)
                object_images = load_object_images(object_image_files)
                carma_processor = Carma(object_images, use_ocad_labels, use_ocad_trigger, model)

                processing_time = time.time()                                
                for image_filename in image_filenames:
                    print(f"loading: {image_filename}")
                    person_actions = {}
                    timestamp = image_filename[:-4]
                    current_image = read_image_as_cv(os.path.join(images_path, image_filename))                    
                    image_quadrupel = carma_processor.add_image(scale_image_cv_max_size(current_image, 512))
                    for person_action_file in person_action_files:
                        if timestamp in person_action_file:
                            person_actions_data = load_pickle(person_action_file)         
                            for person_id, data in person_actions_data.items():
                                if data[-1]["trigger"]:
                                    person_actions.update({person_id: {"image": data[-1]["image"], "action": data[-1]["action"]}})
                    action_patterns = carma_processor.process(image_quadrupel, person_actions, analyze=True)
                    carma_processor.write_result(export_folder, image_filename, action_patterns)

if __name__ == "__main__":


    # ########################## RUNS CONFIGURATION #################################
    # run settings: ["trigger", ""], ["label", ""], ["gpt-4o", "gpt-5", "gemini-2.5-flash", ""]
    run_settings = [("label", "trigger", "gpt-4o")]

    # ########################## BASIC CONTROL #######################################
    show_images = False
    write_results = True
    create_ground_truth = False

    # ########################## EXPERIMENTS #########################################
    iterations = 1
    base_folder = "data"
    experiments = ["scene_020_sf2P", "scene_021_sf2P", "scene_022_sf2P", "scene_026_sf1P1R", "scene_027_sf1P1R", "scene_029_sf2P1R", "scene_0290_sf2P1R",
                   "scene_030_po2P", "scene_032_po2P", "scene_033_po1P1R", "scene_034_po1P1R", "scene_041_ha2P", "scene_042_ha2P", "scene_043_ha1P1R", "scene_044_ha1P1R"]
    # experiments = ["scene_009_PsortO"]

    main(run_settings, experiments, base_folder, show_images, write_results, create_ground_truth, iterations)

