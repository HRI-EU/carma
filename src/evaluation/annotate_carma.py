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
from carma.image_tools.image_tools import show_image_cv, read_image_as_cv, stitch_images, save_image_as_cv

def list_files_and_ids(images_path):
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

def load_result_files(results_path):
    mapping = {}
    for name in os.listdir(results_path):
        if name.endswith(".json"):
            p = os.path.join(results_path, name)
            try:
                with open(p, "r") as f:
                    mapping[name] = json.load(f)
            except Exception:
                mapping[name] = None
    return mapping

def json_name_for(base_name, id_val):
    return f"{base_name}_id_{id_val}.json"


def create_default_set(ground_truth_folder, results_folder, results_files, image_files, person_ids):
    contents = {}
    for file_name in image_files:
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        for id_val in person_ids:
            id_filename = json_name_for(base_name, id_val)
            results_filename = os.path.join(results_folder, id_filename)
            ground_truth_filename = os.path.join(ground_truth_folder, id_filename)
            if id_filename in results_files:
                with open(results_filename, "r") as f:
                    content = json.load(f)
                    contents.update({id_val: content})
            elif id_val in contents:
                content = contents[id_val]
            else:
                content = json.dumps(None)
            with open(ground_truth_filename, "w") as f:
                json.dump(content, f)


def visualize_annotation(ground_truth_folder, images_folder, object_images_folder, person_actions_folder, results_folder):
    object_images = {}
    person_images = {}
    person_action_files = os.listdir(person_actions_folder)
    for object_image_label in os.listdir(object_images_folder):
        object_image = read_image_as_cv(os.path.join(object_images_folder, object_image_label))
        object_images.update({object_image_label[:-4]: object_image})
    image_files, person_ids = list_files_and_ids(images_folder)
    default_ap = [{"object": None, "action": None, "robot_interaction": False}]
    for image_file in image_files:
        ground_truth_text = []
        for person_id in person_ids:
            ground_truth_filename = f"{image_file[:-4]}_id_{person_id}"
            with open(os.path.join(results_folder, ground_truth_filename + ".json"), "w") as f:
                json.dump(default_ap, f)
            with open(os.path.join(ground_truth_folder, ground_truth_filename + ".json"), "r") as f:
                ground_truth = json.load(f)
                if ground_truth == "null":
                    ground_truth = f"{person_id[-4:]}: no action"
                else:
                    if "object" in ground_truth[0]:
                        ground_truth_object = ground_truth[0]["object"]
                    else:
                        ground_truth_object = ""
                    ground_truth_action = ground_truth[0]["action"]
                    ground_truth = f"{person_id[-4:]}: {ground_truth_action} {ground_truth_object}"
                ground_truth_text.append(ground_truth)
            if ground_truth_filename + ".pkl" in person_action_files:
                with open(os.path.join(person_actions_folder, ground_truth_filename + ".pkl"), "rb") as f:
                    person_actions = pickle.load(f)
                    if "id_" + person_id in person_actions:
                        person_image = person_actions["id_" + person_id][0]["image"]
                        person_images.update({f"{person_id[-4:]}": person_image})
        caption_text = [ground_truth_text[0]]
        post_text = None
        if len(ground_truth_text) > 1:
            post_text = ground_truth_text[1]

        image = read_image_as_cv(os.path.join(images_folder, image_file))
        instance_images = list(object_images.values()) + list(person_images.values())
        instance_labels = list(object_images.keys()) + list(person_images.keys())
        stitched_object_images = stitch_images(instance_images, line_offset=10,
                                               caption_text=instance_labels,
                                               font_size=0.4, scale=1.0, grid_size=(2,4))
        stitched_image = stitch_images([image], post_text=post_text, caption_text=caption_text,
                                       scale=0.5, line_offset=10, border_size=0, font_size=0.7)
        stitched_image = stitch_images([stitched_object_images, stitched_image], grid_size=(2,1),
                                       scale=1.0, post_text=f"timestamp: {image_file[:-4]}", font_size=0.5)
        show_image_cv(stitched_image, destroy_all_windows=False, wait_key=1)
        # save_image_as_cv(stitched_image, image_file)
        # Wait for key press
        print("Press [any key] next or [e] edit label")
        key = cv2.waitKey(0) & 0xFF

        if key == ord('e'):
            for person_id in person_ids:
                ground_truth_filename = f"{image_file[:-4]}_id_{person_id}"
                with open(os.path.join(ground_truth_folder, ground_truth_filename + ".json"), "r") as f:
                    ground_truth = json.load(f)
                    for _type in ["action", "object"]:
                        label = input(f"{person_id[-4:]} {_type}: ")
                        if label == "":
                            label = None
                        if len(ground_truth) > 0 and "action" in ground_truth[0] and "object" in ground_truth[0]:
                            ground_truth[0][_type] = label
                    with open(os.path.join(results_folder, ground_truth_filename + ".json"), "w") as f:
                        json.dump(default_ap, f)


if __name__ == "__main__":
    base_folder = "./data/scene_021_sf2P"
    images_folder = os.path.join(base_folder, "images")
    object_images_folder = os.path.join(base_folder, "object_images")
    ground_truth_folder = os.path.join(base_folder, "ground_truth")
    person_actions_folder = os.path.join(base_folder, "person_actions")
    results_folder = os.path.join(base_folder, "results")

    image_files, ids = list_files_and_ids(images_folder)
    result_files = load_result_files(results_folder)
    # create_default_set(ground_truth_folder, results_folder, result_files, image_files, ids)
    visualize_annotation(ground_truth_folder, images_folder,
                         object_images_folder, person_actions_folder, results_folder)
