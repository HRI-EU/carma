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

from carma.instance_clusterer.instance_clusterer import InstanceClusterer
from carma.instance_detector.instance_detector import InstanceDetector
from carma.object_detector.owlvi_detector import OWLViTDetector
from carma.image_tools.image_tools import (
    save_image_as_cv,
    read_image_as_cv,
    show_image_cv,
    crop_rois,
    stitch_images,
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
    def __init__(self, object_images, use_ocad_labels=True, use_ocad_trigger=False, prev_action=True):
        self.similarity_threshold = 0.0
        self.frame_count = 0
        self.prev_action = prev_action
        self.use_ocad_labels = use_ocad_labels
        self.use_ocad_trigger = use_ocad_trigger
        self.object_images = object_images
        self.instance_clusterer = InstanceClusterer(distance_threshold=self.similarity_threshold)
        self.instance_clusterer.create_clusters(list(self.object_images.values()), list(self.object_images.keys()))
        self.instance_detector = InstanceDetector(vlm_model="gpt4")
        self.object_detector = OWLViTDetector(conf=0.007)
        self.previous_actions = {}
        self.image_buffer = {}
        self.instance_detector.pre_text = (
            "You are given a sequence of images. The first images serve as references for labeling objects. "
            "Each reference image has a caption in the form 'object_x' for labeled object images, where 'x' is the "
            "ID of the object. For image sequences, labels are 'image_y', where 'y' is the frame number. "
            "Your task:"
            "- Look only at the final image to detect the action of a person and the object it interacts with. "
            "Take the captioning of the final images into account - if present." 
            "- Double check if the actor really touches the object with his hands and interacts with it. "
            "If not, use the label 'idle'. Exclusively use following atomic actions: grasp, handover, place_down, "
            "hold or pour.\n"
            "Important:"
            "1. Focus on a single action, person relation, considering on the previous action provided in the "
            "image label.\n"
            "2. If you identify the robot_hand of the robot in the image, verify if the robot is interacting with the "
            "human. If yes, set the item {'robot_interaction': true} if not, set it to {'robot_interaction':false}. "
            "3. You must include the spatial relation to a second involved object if both objects are placed in or on "
            "each other, like {'object': 'object_1', 'action': 'place_down', 'on': 'object_3'}. The objects must be "
            "obviously in contact.\n"
        )
        self.instance_detector.post_text = (
            "Return a JSON a dictionary describing the action of the human actor like {'object': 'object_2', "
            "'action': 'hold', 'robot_interaction': False}."
        )

    def create_action_patterns(self, action_images, action_captions, object_images, object_captions):
        images = action_images + object_images
        response = self.instance_detector.identify_instances(
            images, image_captions=action_captions + object_captions, response_format="json_object"
        )
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

    def cluster_instances(self, action_image):
        object_images = []
        object_captions = []
        object_confidences = []
        self.instance_clusterer.filter_clusters(keep_first_n=1)
        object_rois = self.object_detector.detect_objects(action_image, exclude_class_names=["person"])
        # show_image_cv(draw_rois(action_image, object_rois, show_labels=False), wait_key=0)
        cropped_images = crop_rois(action_image, object_rois)
        assignments = self.instance_clusterer.add_instances(list(cropped_images.values()), create_new_clusters=False)
        for image_index in range(len(cropped_images)):
            if assignments[image_index]:
                object_caption = assignments[image_index][0]
                object_confidence = assignments[image_index][1]
                object_captions.append(object_caption)
                object_confidences.append(object_confidence)
                object_images.append(self.object_images[object_caption])
        return object_images, object_captions, object_confidences

    def process(self, frame, analyze=True):
        responses = []
        stitched_images = []
        for person_id, actions_msgs in frame.items():
            action_captions = []
            image_captions = []
            action_images = []
            action_timestamps = []
            trigger = False if self.use_ocad_trigger else True
            nth_image = 1 if self.use_ocad_trigger else 4
            sorted_images = []
            for actions_msg in actions_msgs:
                if "action" in actions_msg:
                    action_captions.append(actions_msg["action"])
                    action_images.append(actions_msg["image"])
                    action_timestamps.append(actions_msg["timestamp"])
                    if actions_msg["trigger"] is True:
                        trigger = True
            for timestamp, image in zip(action_timestamps, action_images):
                if person_id not in self.image_buffer:
                    self.image_buffer[person_id] = {}
                self.image_buffer[person_id][timestamp] = image
            sorted_entries = sorted(self.image_buffer[person_id].items(), key=lambda item: item[0])
            for image_timestamp, image in sorted_entries[-4:]:
                sorted_images.append(image)
            if (len(action_captions) > 0) and (trigger is True) and (self.frame_count % nth_image == 0):
                if self.similarity_threshold > 0:
                    object_images, object_captions, object_confidences = self.cluster_instances(action_images[-1])
                    object_captions = [f"{caption}" for caption, confidence in
                                       zip(object_captions, object_confidences)]
                else:
                    object_images = list(self.object_images.values())
                    object_captions = list(self.object_images.keys())
                action_images = sorted_images
                for img_ix in range(len(action_images)):
                    image_label = f"image_{img_ix}"
                    image_captions.append(image_label)
                previous_action = {} if person_id not in self.previous_actions else self.previous_actions[person_id]
                if previous_action:
                    previous_action_str = (self.previous_actions[person_id]["action"] + " " +
                                           self.previous_actions[person_id]["object"])
                else:
                    previous_action_str = "idle"
                print(f"previous action: {previous_action}")
                if self.use_ocad_labels:
                    image_caption = image_captions[-1]
                    ocad_label = action_captions[-1]
                    image_caption = (f"{image_caption}: person currently {ocad_label} and "
                                     f"his previous action was {previous_action_str}")
                    image_captions[-1] = image_caption
                    action_captions = image_captions
                else:
                    image_caption = image_captions[-1]
                    image_caption = f"{image_caption}: the previous action of the person was {previous_action_str}"
                    image_captions[-1] = image_caption
                    action_captions = image_captions
                if analyze:
                    action_patterns = self.create_action_patterns(action_images, action_captions, object_images,
                                                                  object_captions)
                else:
                    action_patterns = {}
                stitched_image = stitch_images(object_images + action_images, font_size=0.3,
                                               caption_text=object_captions + action_captions, width=256)
                stitched_images.append(stitched_image)
                if action_patterns and "action" in action_patterns:
                    if action_patterns != previous_action:
                        responses.append(action_patterns)
                        self.previous_actions.update({person_id: action_patterns})
                        print("response:", action_patterns)
                    else:
                        print("skipped response:", action_patterns)
                else:
                    print("skipped response:", action_patterns)

        self.frame_count += 1
        return responses, stitched_images


def main(run_settings, runs, base_folder, show_images, write_results, create_ground_truth, iterations):

    # ########################## MAIN LOOP ############################################
    for run in runs:
        for iteration in range(iterations):
            for run_setting in run_settings:
                use_ocad_labels = True if "label" in run_setting[0] else False
                use_ocad_trigger = True if "trigger" in run_setting[1] else False
                prev_action = run_setting[2]
                data_path = os.path.join(base_folder, run)
                run_name = f"{run_setting[0]}-{run_setting[1]}-{run_setting[2]}-{iteration}"
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
                carma_processor = Carma(object_images, use_ocad_labels, use_ocad_trigger, prev_action)

                for person_actions_file in person_action_files:
                    print(carma_processor.frame_count, "/", len(person_action_files))
                    person_actions = load_pickle(person_actions_file)
                    responses, stitched_images = carma_processor.process(person_actions, analyze=True)
                    if responses and write_results:
                        export_pkl = os.path.join(export_folder, f"{os.path.basename(person_actions_file)[:-4]}.json")
                        # export_jpg = os.path.join(export_folder, f"{os.path.basename(person_actions_file)[:-4]}.jpg")
                        export_results(export_pkl, responses)
                    for stitched_image in stitched_images:
                        if show_images:
                            show_image_cv(stitched_image, wait_key=0)
                        # if write_results and responses:
                            # save_image_as_cv(stitched_image, export_jpg)



if __name__ == "__main__":


    # ########################## RUNS CONFIGURATION #################################
    # run settings: [label, ""], ["trigger", ""], ["previous", ""]
    run_settings = [("label", "trigger", "")]

    # ########################## BASIC CONTROL #######################################
    show_images = False
    write_results = True
    create_ground_truth = False

    # ########################## EXPERIMENTS #########################################
    iterations = 1
    base_folder = "data"
    experiments = ["scene_009_PsortO", "scene_020_sf2P", "scene_021_sf2P", "scene_022_sf2P", "scene_026_sf1P1R", "scene_027_sf1P1R", "scene_029_sf2P1R", "scene_0290_sf2P1R",
                   "scene_030_po2P", "scene_032_po2P", "scene_033_po1P1R", "scene_034_po1P1R", "scene_041_ha2P", "scene_042_ha2P", "scene_043_ha1P1R", "scene_044_ha1P1R"]

    main(run_settings, experiments, base_folder, show_images, write_results, create_ground_truth, iterations)
