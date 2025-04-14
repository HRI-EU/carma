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


def evaluate_folders(ground_truth_folder, prediction_folder):
    """
    Evaluate matching between ground-truth and prediction JSON files, matching them
    by filename (exact match). If a file exists in only one folder, its entries are
    compared against an empty list.

    For each file (from the union of filenames in both folders), the entries are compared
    index-by-index (i.e. the first entry in the ground truth is compared with the first entry
    in the prediction, and so on). If one file has more entries than the other, the missing
    entries are treated as errors.

    For each compared entry:
      - An overall (triplet) match is counted if the person, action, and object fields all match.
      - Additionally, individual matches for person, action, and object are counted.

    Metrics for each (overall, person, action, object) are computed as:
      - Precision = TP / total_pred  (if total_pred > 0)
      - Recall    = TP / total_gt    (if total_gt > 0)
      - Accuracy  = TP / total_indices (if total_indices > 0)

    Here, for each file:
      - total_gt is the number of ground truth entries.
      - total_pred is the number of prediction entries.
      - total_indices is the maximum of the two lengths.

    :param ground_truth_folder: Folder containing ground-truth JSON files.
    :param prediction_folder: Folder containing prediction JSON files.
    :return: A dictionary with metrics for overall, person, action, and object.
    """
    # Initialize overall counters
    TP_overall = 0
    total_gt_overall = 0
    total_pred_overall = 0
    total_indices_overall = 0

    # Initialize individual field counters
    TP_person = 0
    TP_action = 0
    TP_object = 0
    TP_collaboration = 0
    total_gt_person = 0
    total_pred_person = 0
    total_indices_person = 0

    total_gt_action = 0
    total_pred_action = 0
    total_indices_action = 0

    total_gt_object = 0
    total_pred_object = 0
    total_indices_object = 0

    total_gt_collaboration = 0
    total_pred_collaboration = 0
    total_indices_collaboration = 0

    # Get the union of filenames from both folders (only JSON files)
    gt_files = {f for f in os.listdir(ground_truth_folder) if f.endswith(".json")}
    pred_files = {f for f in os.listdir(prediction_folder) if f.endswith(".json")}
    all_files = sorted(gt_files.union(pred_files))

    # Process each file in the union
    for filename in all_files:
        gt_path = os.path.join(ground_truth_folder, filename)
        pred_path = os.path.join(prediction_folder, filename)

        # Load ground-truth entries if file exists; else use empty list.
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                gt_entries = json.load(f)
        else:
            gt_entries = []

        # Load prediction entries if file exists; else use empty list.
        if os.path.exists(pred_path):
            with open(pred_path, "r") as f:
                pred_entries = json.load(f)
        else:
            pred_entries = []

        # Determine number of positions to compare (max of two lengths)
        n = max(len(gt_entries), len(pred_entries))
        total_indices_overall += n
        total_indices_person += n
        total_indices_action += n
        total_indices_object += n

        # Count ground-truth and prediction entries
        total_gt_overall += len(gt_entries)
        total_pred_overall += len(pred_entries)
        total_gt_person += len(gt_entries)
        total_pred_person += len(pred_entries)
        total_gt_action += len(gt_entries)
        total_pred_action += len(pred_entries)
        total_gt_object += len(gt_entries)
        total_pred_object += len(pred_entries)
        total_gt_collaboration += len(gt_entries)
        total_pred_collaboration += len(pred_entries)

        # Compare entries index-by-index
        for i in range(n):
            gt_entry = gt_entries[i] if i < len(gt_entries) else None
            pred_entry = pred_entries[i] if i < len(pred_entries) else None

            # Only compare if both entries exist at this index.
            if gt_entry is not None and pred_entry is not None:
                gt_person = str(gt_entry.get("person", ""))
                pred_person = str(pred_entry.get("person", ""))
                gt_action = str(gt_entry.get("action", ""))
                pred_action = str(pred_entry.get("action", ""))
                gt_object = str(gt_entry.get("object", ""))
                pred_object = str(pred_entry.get("object", ""))
                gt_collaboration = str(gt_entry.get("robot_interaction", ""))
                pred_collaboration = str(pred_entry.get("robot_interaction", ""))

                # Overall triplet match
                if (gt_person == pred_person and gt_action == pred_action and gt_object == pred_object and
                        gt_collaboration == pred_collaboration):
                    TP_overall += 1

                # Individual field comparisons
                if gt_person == pred_person:
                    TP_person += 1
                if gt_action == pred_action:
                    TP_action += 1
                if gt_object == pred_object:
                    TP_object += 1
                if gt_collaboration == pred_collaboration:
                    TP_collaboration += 1
            # If one entry is missing at this index, no match is counted.

    # Compute metrics for overall triplet matching
    precision_overall = TP_overall / total_pred_overall if total_pred_overall > 0 else 0.0
    recall_overall = TP_overall / total_gt_overall if total_gt_overall > 0 else 0.0
    accuracy_overall = TP_overall / total_indices_overall if total_indices_overall > 0 else 0.0

    # Compute metrics for individual fields
    precision_person = TP_person / total_pred_person if total_pred_person > 0 else 0.0
    recall_person = TP_person / total_gt_person if total_gt_person > 0 else 0.0
    accuracy_person = TP_person / total_indices_person if total_indices_person > 0 else 0.0

    precision_action = TP_action / total_pred_action if total_pred_action > 0 else 0.0
    recall_action = TP_action / total_gt_action if total_gt_action > 0 else 0.0
    accuracy_action = TP_action / total_indices_action if total_indices_action > 0 else 0.0

    precision_object = TP_object / total_pred_object if total_pred_object > 0 else 0.0
    recall_object = TP_object / total_gt_object if total_gt_object > 0 else 0.0
    accuracy_object = TP_object / total_indices_object if total_indices_object > 0 else 0.0

    precision_collaboration = TP_collaboration / total_pred_collaboration if total_pred_collaboration > 0 else 0.0
    recall_collaboration = TP_collaboration / total_gt_collaboration if total_gt_collaboration > 0 else 0.0
    accuracy_collaboration = TP_collaboration / total_indices_collaboration if total_indices_collaboration > 0 else 0.0

    results = {
        "overall": {
            "TP": TP_overall,
            "total_gt": total_gt_overall,
            "total_pred": total_pred_overall,
            "total_indices": total_indices_overall,
            "precision": precision_overall,
            "recall": recall_overall,
            "accuracy": accuracy_overall,
        },
        "person": {
            "TP": TP_person,
            "total_gt": total_gt_person,
            "total_pred": total_pred_person,
            "total_indices": total_indices_person,
            "precision": precision_person,
            "recall": recall_person,
            "accuracy": accuracy_person,
        },
        "action": {
            "TP": TP_action,
            "total_gt": total_gt_action,
            "total_pred": total_pred_action,
            "total_indices": total_indices_action,
            "precision": precision_action,
            "recall": recall_action,
            "accuracy": accuracy_action,
        },
        "object": {
            "TP": TP_object,
            "total_gt": total_gt_object,
            "total_pred": total_pred_object,
            "total_indices": total_indices_object,
            "precision": precision_object,
            "recall": recall_object,
            "accuracy": accuracy_object,
        },
        "collaboration": {
            "TP": TP_collaboration,
            "total_gt": total_gt_collaboration,
            "total_pred": total_pred_collaboration,
            "total_indices": total_indices_collaboration,
            "precision": precision_collaboration,
            "recall": recall_collaboration,
            "accuracy": accuracy_collaboration,
        },
    }
    return results

def main(run_settings, runs):

    # ########################## MAIN LOOP ############################################
    all_metrics = {
        "overall": [],
        "person": [],
        "action": [],
        "object": [],
        "collaboration": []
    }

    # Loop over all runs and run settings
    for run in runs:
        for run_setting in run_settings:
            use_ocad_labels = run_setting[0]
            use_ocad_trigger = run_setting[1]
            prev_action = run_setting[2]
            run_folder = f"{use_ocad_labels}-{use_ocad_trigger}-{prev_action}"
            ground_truth_folder = f"data/{run}/ground_truth/{run_folder}/"
            prediction_folder = f"data/{run}/runs/{run_folder}"

            # Get metrics for the current run
            metrics = evaluate_folders(ground_truth_folder, prediction_folder)

            # Print individual run metrics as before
            print("Overall Metrics:")
            print(f"  True Positives   : {metrics['overall']['TP']}")
            print(f"  Total GroundTruth: {metrics['overall']['total_gt']}")
            print(f"  Total Predictions: {metrics['overall']['total_pred']}")
            print(f"  Total Indices    : {metrics['overall']['total_indices']}")
            print(f"  Precision        : {metrics['overall']['precision']:.3f}")
            print(f"  Recall           : {metrics['overall']['recall']:.3f}")
            print(f"  Accuracy         : {metrics['overall']['accuracy']:.3f}\n")

            print("Person Metrics:")
            print(f"  True Positives   : {metrics['person']['TP']}")
            print(f"  Total GroundTruth: {metrics['person']['total_gt']}")
            print(f"  Total Predictions: {metrics['person']['total_pred']}")
            print(f"  Precision        : {metrics['person']['precision']:.3f}")
            print(f"  Recall           : {metrics['person']['recall']:.3f}")
            print(f"  Accuracy         : {metrics['person']['accuracy']:.3f}\n")

            print("Action Metrics:")
            print(f"  True Positives   : {metrics['action']['TP']}")
            print(f"  Total GroundTruth: {metrics['action']['total_gt']}")
            print(f"  Total Predictions: {metrics['action']['total_pred']}")
            print(f"  Precision        : {metrics['action']['precision']:.3f}")
            print(f"  Recall           : {metrics['action']['recall']:.3f}")
            print(f"  Accuracy         : {metrics['action']['accuracy']:.3f}\n")

            print("Object Metrics:")
            print(f"  True Positives   : {metrics['object']['TP']}")
            print(f"  Total GroundTruth: {metrics['object']['total_gt']}")
            print(f"  Total Predictions: {metrics['object']['total_pred']}")
            print(f"  Precision        : {metrics['object']['precision']:.3f}")
            print(f"  Recall           : {metrics['object']['recall']:.3f}")
            print(f"  Accuracy         : {metrics['object']['accuracy']:.3f}\n")

            print("Collaboration Metrics:")
            print(f"  True Positives   : {metrics['collaboration']['TP']}")
            print(f"  Total GroundTruth: {metrics['collaboration']['total_gt']}")
            print(f"  Total Predictions: {metrics['collaboration']['total_pred']}")
            print(f"  Precision        : {metrics['collaboration']['precision']:.3f}")
            print(f"  Recall           : {metrics['collaboration']['recall']:.3f}")
            print(f"  Accuracy         : {metrics['collaboration']['accuracy']:.3f}\n")

            # Save each run's metrics for later averaging
            for category in all_metrics:
                all_metrics[category].append(metrics[category])

    # Helper function to compute the average of each metric in a list of metric dictionaries
    def average_metrics(metrics_list):
        avg = {}
        count = len(metrics_list)
        if count == 0:
            return avg
        # Assume each dictionary in the list has the same keys
        for key in metrics_list[0].keys():
            total = sum(m[key] for m in metrics_list)
            avg[key] = total / count
        return avg

    # Compute averaged metrics for each category
    averaged_metrics = {}
    for category, metrics_list in all_metrics.items():
        averaged_metrics[category] = average_metrics(metrics_list)

    # Print the aggregated averaged metrics in the same format as the individual ones
    print("Aggregated Averaged Metrics Across All Runs:\n")

    print("Overall Metrics:")
    print(f"  True Positives   : {averaged_metrics['overall']['TP']:.2f}")
    print(f"  Total GroundTruth: {averaged_metrics['overall']['total_gt']:.2f}")
    print(f"  Total Predictions: {averaged_metrics['overall']['total_pred']:.2f}")
    print(f"  Total Indices    : {averaged_metrics['overall']['total_indices']:.2f}")
    print(f"  Precision        : {averaged_metrics['overall']['precision']:.3f}")
    print(f"  Recall           : {averaged_metrics['overall']['recall']:.3f}")
    print(f"  Accuracy         : {averaged_metrics['overall']['accuracy']:.3f}\n")

    print("Person Metrics:")
    print(f"  True Positives   : {averaged_metrics['person']['TP']:.2f}")
    print(f"  Total GroundTruth: {averaged_metrics['person']['total_gt']:.2f}")
    print(f"  Total Predictions: {averaged_metrics['person']['total_pred']:.2f}")
    print(f"  Precision        : {averaged_metrics['person']['precision']:.3f}")
    print(f"  Recall           : {averaged_metrics['person']['recall']:.3f}")
    print(f"  Accuracy         : {averaged_metrics['person']['accuracy']:.3f}\n")

    print("Action Metrics:")
    print(f"  True Positives   : {averaged_metrics['action']['TP']:.2f}")
    print(f"  Total GroundTruth: {averaged_metrics['action']['total_gt']:.2f}")
    print(f"  Total Predictions: {averaged_metrics['action']['total_pred']:.2f}")
    print(f"  Precision        : {averaged_metrics['action']['precision']:.3f}")
    print(f"  Recall           : {averaged_metrics['action']['recall']:.3f}")
    print(f"  Accuracy         : {averaged_metrics['action']['accuracy']:.3f}\n")

    print("Object Metrics:")
    print(f"  True Positives   : {averaged_metrics['object']['TP']:.2f}")
    print(f"  Total GroundTruth: {averaged_metrics['object']['total_gt']:.2f}")
    print(f"  Total Predictions: {averaged_metrics['object']['total_pred']:.2f}")
    print(f"  Precision        : {averaged_metrics['object']['precision']:.3f}")
    print(f"  Recall           : {averaged_metrics['object']['recall']:.3f}")
    print(f"  Accuracy         : {averaged_metrics['object']['accuracy']:.3f}\n")

    print("Collaboration Metrics:")
    print(f"  True Positives   : {averaged_metrics['collaboration']['TP']:.2f}")
    print(f"  Total GroundTruth: {averaged_metrics['collaboration']['total_gt']:.2f}")
    print(f"  Total Predictions: {averaged_metrics['collaboration']['total_pred']:.2f}")
    print(f"  Precision        : {averaged_metrics['collaboration']['precision']:.3f}")
    print(f"  Recall           : {averaged_metrics['collaboration']['recall']:.3f}")
    print(f"  Accuracy         : {averaged_metrics['collaboration']['accuracy']:.3f}\n")


if __name__ == "__main__":

    # ########################## RUNS CONFIGURATION #################################
    # run setting: label, trigger, previous action
    run_settings = [(False, True, False)]

    # ########################## EXPERIMENTS #########################################
    experiments = {"sorting_fruits": ["scene_009_PsortO", "scene_020_sf2P", "scene_021_sf2P", "scene_022_sf2P",
                                      "scene_026_sf1P1R", "scene_027_sf1P1R", "scene_029_sf2P1R", "scene_0290_sf2P1R"],
                   "pouring": ["scene_030_po2P", "scene_032_po2P", "scene_033_po1P1R", "scene_034_po1P1R",
                               "scene_035_po1P1R"],
                   "handover": ["scene_041_ha2P", "scene_042_ha2P", "scene_043_ha1P1R", "scene_044_ha1P1R"]
                   }
    runs = experiments["sorting_fruits"][:1]
    main(run_settings, runs)
