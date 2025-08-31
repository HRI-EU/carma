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
import json
from pathlib import Path


def evaluate_folders(ground_truth_folder, prediction_folder):
    total_pred_action_pattern = 0
    pred_files = sorted([f for f in os.listdir(prediction_folder) if f.endswith(".json")])
    gt_range = 20  # search 1s back and forth
    for pred_file in pred_files:
        file_pattern = pred_file.split("_id_")[-1]
        gt_matches = Path(ground_truth_folder).glob("*" + file_pattern)
        gt_matches = [f.name for f in gt_matches]
        if pred_file in gt_matches:
            gt_ix = gt_matches.index(pred_file)
            lower_ix_bound = 0 if gt_ix < gt_range else gt_ix - gt_range
            upper_ix_bound = gt_ix if (gt_ix + gt_range) > len(gt_matches) else gt_ix + gt_range
            gt_entries = []
            pred_entry = None
            for ix in range(lower_ix_bound, upper_ix_bound):
                gt_path = os.path.join(ground_truth_folder, gt_matches[ix])
                pred_path = os.path.join(prediction_folder, pred_file)
                with open(gt_path, "r") as f:
                    gt_entry = json.load(f)
                    gt_action = str(gt_entry[0]["action"]).strip()
                    gt_object = str(gt_entry[0]["object"]).strip()
                    gt_on = str(gt_entry[0]["on"]).strip()
                    robot_interaction = str(gt_entry[0]["robot_interaction"]).strip()
                    gt_entry = gt_action + " " + gt_object + " " + gt_on + " " + robot_interaction
                    gt_entries.append(gt_entry)
                with open(pred_path, "r") as f:                    
                    pred_entry = json.load(f)
                    print(pred_entry)
                    pred_entry = pred_entry[0]
            print("---------------------------------------------------------------")
            print(gt_entries)
            gt_entries = list(set(gt_entries))
            if pred_entry and len(gt_entries) > 0:
                print(pred_entry)
                action_synonyms = {"hold": ["pick_up", "grasp"], "place_down": ["place"], "idle": ["idle"],
                                   "grasp": ["pick_up", "hold"], "pour": ["fill"], "handover": ["hold"]}
                # create all synonym candidates
                print(pred_entry)
                pred_actions = action_synonyms[pred_entry["action"]] + [pred_entry["action"]]
                on_string = "" if "on" not in pred_entry else pred_entry["on"]
                for pred_action in pred_actions:
                    action_pattern = (pred_action.strip() + " " + pred_entry["object"].strip() + " " + on_string.strip()
                                      + " " + str(pred_entry["robot_interaction"]).strip())
                    print(action_pattern)
                    if action_pattern in gt_entries:
                        total_pred_action_pattern += 1
                        print("match")
                        break

    results = {"action_patterns": total_pred_action_pattern/len(pred_files)}

    return results


def main(run_settings, runs):
    overall_metrics = {}
    for run in runs:
        for run_setting in run_settings:
            if run_setting in ["gpt4"]:
                run_folder = run_setting
            else:
                use_ocad_labels = run_setting[0]
                use_ocad_trigger = run_setting[1]
                prev_action = run_setting[2]
                run_folder = f"{use_ocad_labels}-{use_ocad_trigger}-{prev_action}"
            ground_truth_folder = f"/hri/localdisk/deigmoel/data_icra/{run}/ground_truth/"
            prediction_folder = f"/hri/localdisk/deigmoel/data_icra/{run}/runs/{run_folder}"

            # Get metrics for the current run
            metrics = evaluate_folders(ground_truth_folder, prediction_folder)
            overall_metrics.update({f"{run}_{run_folder}": metrics})
    averages = {}
    for run, results in overall_metrics.items():
        run_type = run.split("_")[-1]
        if run_type in averages:
            averages[run_type].append(results["action_patterns"])
        else:
            averages.update({run_type: [results["action_patterns"]]})
    for run_type, values in averages.items():
        average = sum(values) / len(values)
        print(run_type, average)


if __name__ == "__main__":

    # ########################## RUNS CONFIGURATION #################################
    # run setting: label, trigger, previous action
    run_settings = [(False, False, False)]
    run_settings = [("gpt4")]

    # ########################## EXPERIMENTS #########################################
    experiments = {"sorting_fruits": ["scene_009_PsortO", "scene_020_sf2P", "scene_021_sf2P", "scene_022_sf2P",
                                      "scene_026_sf1P1R", "scene_027_sf1P1R", "scene_029_sf2P1R", "scene_0290_sf2P1R"],
                   "pouring": ["scene_030_po2P", "scene_032_po2P", "scene_033_po1P1R", "scene_034_po1P1R",
                               "scene_035_po1P1R"],
                   "handover": ["scene_041_ha2P", "scene_042_ha2P", "scene_043_ha1P1R", "scene_044_ha1P1R"]
                   }
    # scene_035_po1P1R images are missing

    # runs = experiments["sorting_fruits"][0:1]  # scene_009_PsortO
    runs = experiments["sorting_fruits"] + experiments["pouring"] + experiments["handover"]
    runs = ["scene_009_PsortO"]
    main(run_settings, runs)
