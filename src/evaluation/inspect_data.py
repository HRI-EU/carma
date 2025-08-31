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

def list_files_and_ids(images_path, person_actions_folder):
    pattern_no_id = re.compile(r"^\d+\.\d+\.jpg$")
    pattern_id = re.compile(r"_id_([^.]+)\.jpg$")
    image_files = []
    pkl_files = []
    unique_ids = set()
    for file_name in os.listdir(images_path):
        if pattern_no_id.match(file_name):
            image_files.append(file_name)
        else:
            m = pattern_id.search(file_name)
            if m:
                unique_ids.add(m.group(1))
    image_files = sorted(image_files)
    unique_ids = sorted(list(unique_ids))
    for file_name in os.listdir(person_actions_folder):
        if file_name.endswith(".pkl"):
            pkl_files.append(os.path.join(person_actions_folder, file_name))
    pkl_files = sorted(pkl_files)
    return image_files, unique_ids, pkl_files

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

def visualize_data(image_files, pkl_files):
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as f:
            person_actions = pickle.load(f)  
            for person_id, person_data in person_actions.items():
                for data_entry in person_data:
                    show_image_cv(data_entry["image"], destroy_all_windows=False, wait_key=0)


if __name__ == "__main__":
    base_folder = "/hri/localdisk/deigmoel/data_icra/scene_009_PsortO"
    images_folder = os.path.join(base_folder, "images")
    object_images_folder = os.path.join(base_folder, "object_images")
    ground_truth_folder = os.path.join(base_folder, "ground_truth")
    person_actions_folder = os.path.join(base_folder, "person_actions")
    results_folder = os.path.join(base_folder, "results")

    # get all files and person_ids
    image_files, ids, pkl_files = list_files_and_ids(images_folder, person_actions_folder)
    visualize_data(image_files, pkl_files)
    
