import os
import json
import glob

action_synonyms = {
    "hold": ["pick_up", "grasp"],
    "place_down": ["place"],
    "idle": ["idle"],
    "grasp": ["pick_up", "hold"],
    "pick_up": ["grasp", "hold"],
    "pour": ["fill"],
    "handover": ["hold"],
}

def load_ground_truth(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_measurements(folder):
    data = {}
    for filename in os.listdir(folder):
        if not filename.endswith(".json") or "_id_" not in filename:
            continue
        ts_str, rest = filename.split("_id_", 1)
        file_id = rest.replace(".json", "").strip()
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        data.setdefault(file_id, {})[ts_str] = obj
    return data

def get_processing_stats(processing_time_path):
    if not os.path.isfile(processing_time_path):
        return 0.0, 0
    with open(processing_time_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    pt = float(data.get('processing_time', 0.0))
    images = int(data.get('images', 0))
    if images <= 0 or pt <= 0.0:
        return 0.0, 0
    return pt, images

def normalize_action(action):
    for k, synonyms in action_synonyms.items():
        if action == k or action in synonyms:
            return k
    return action

def event_key(evt_list):
    if not isinstance(evt_list, list) or not evt_list or not isinstance(evt_list[0], dict):
        return ("", "", "", False)
    d = evt_list[0]
    return (
        normalize_action(d.get("action", "")),
        d.get("object", ""),
        d.get("on", ""),
        bool(d.get("robot_interaction", False)),
    )

def strip_idle_from_gt(gt):
    out = {}
    for pid, ts_map in gt.items():
        kept = {}
        for ts, evt_list in ts_map.items():
            if isinstance(evt_list, list) and evt_list and isinstance(evt_list[0], dict):
                if evt_list[0].get("action") == "idle":
                    continue
            kept[ts] = evt_list
        if kept:
            out[pid] = kept
    return out

def evaluate_run(gt, meas, tolerance_s=1.0):
    ids = sorted(set(gt.keys()) | set(meas.keys()))
    total_gt = 0
    correct_full = 0
    correct_action = 0
    correct_object = 0
    correct_on = 0
    correct_robot = 0
    total_meas = sum(len(meas.get(pid, {})) for pid in ids)
    used_global = 0

    for pid in ids:
        gt_items = sorted(((float(ts), evt) for ts, evt in gt.get(pid, {}).items()), key=lambda x: x[0])
        meas_items = sorted(((float(ts), evt) for ts, evt in meas.get(pid, {}).items()), key=lambda x: x[0])
        used = [False] * len(meas_items)

        for tg, evg in gt_items:
            total_gt += 1
            key_g = event_key(evg)
            act_g, obj_g, on_g, robot_g = key_g

            candidates = [i for i, (tm, em) in enumerate(meas_items)
                          if not used[i] and abs(tm - tg) <= tolerance_s]

            matched_action = matched_object = matched_on = matched_robot = False
            for i in candidates:
                a_m, o_m, on_m, r_m = event_key(meas_items[i][1])
                if a_m == act_g:
                    matched_action = True
                if o_m == obj_g:
                    matched_object = True
                if on_m == on_g:
                    matched_on = True
                if r_m == robot_g:
                    matched_robot = True
                if matched_action and matched_object and matched_on and matched_robot:
                    break

            if matched_action:
                correct_action += 1
            if matched_object:
                correct_object += 1
            if matched_on:
                correct_on += 1
            if matched_robot:
                correct_robot += 1

            full_candidates = [i for i in candidates if event_key(meas_items[i][1]) == key_g]
            if full_candidates:
                best_i = min(full_candidates, key=lambda i: abs(meas_items[i][0] - tg))
                used[best_i] = True
                used_global += 1
                correct_full += 1

    TP = correct_full
    FP = max(total_meas - used_global, 0)
    FN = max(total_gt - TP, 0)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    acc_full   = correct_full   / total_gt if total_gt else 0.0
    acc_action = correct_action / total_gt if total_gt else 0.0
    acc_object = correct_object / total_gt if total_gt else 0.0
    acc_on     = correct_on     / total_gt if total_gt else 0.0
    acc_robot  = correct_robot  / total_gt if total_gt else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN,
        "precision": precision, "recall": recall, "f1": f1,
        "total_gt": total_gt, "total_meas": total_meas,
        "acc_full": acc_full,
        "acc_action": acc_action,
        "acc_object": acc_object,
        "acc_on": acc_on,
        "acc_robot": acc_robot
    }

if __name__ == "__main__":
    experiments_groups = [
        ["scene_009_PsortO",
        "scene_020_sf2P", "scene_021_sf2P", "scene_022_sf2P",
        "scene_026_sf1P1R", "scene_027_sf1P1R",
        "scene_029_sf2P1R", "scene_0290_sf2P1R"],
        ["scene_030_po2P", "scene_032_po2P",
        "scene_033_po1P1R", "scene_034_po1P1R"],
        ["scene_041_ha2P", "scene_042_ha2P",
        "scene_043_ha1P1R", "scene_044_ha1P1R"]
    ]

    model = "gpt-5-0"
    model = "trigger-label-gpt-5-0"
    model = "gemini-2.5-flash-0"
    model = "trigger-label-gemini-2.5-flash-0"
    model = "gpt-4o-0"
    model = "trigger-label-gpt-4o-0"
    tolerance_s = 5.0

    global_TP = global_FP = global_FN = 0
    global_total_gt = global_total_meas = 0
    global_acc_full = global_acc_action = global_acc_object = global_acc_on = global_acc_robot = 0
    global_runs = 0

    total_pt_seconds = 0.0
    total_pt_images = 0

    group_f1_list = []
    group_action_avg = []
    group_object_avg = []
    group_spatial_avg = []
    group_robot_avg = []

    for gi, group in enumerate(experiments_groups, start=1):
        group_TP = group_FP = group_FN = 0
        group_total_gt = group_total_meas = 0
        group_acc_full = group_acc_action = group_acc_object = group_acc_on = group_acc_robot = 0
        group_runs = 0
        print("\n" + "="*50)
        print(f"GROUP {gi}: {group}")
        for experiment in group:
            folder_pattern = f"data/{experiment}/runs/{model}"
            meas_folders = glob.glob(folder_pattern + "*")
            for meas_folder in meas_folders:
                gt_path = f"data/{experiment}/ground_truth.json"
                processing_time_path = f"{meas_folder}/processing_time.json"
                ground_truth = strip_idle_from_gt(load_ground_truth(gt_path))
                measurements = load_measurements(meas_folder)
                res = evaluate_run(ground_truth, measurements, tolerance_s=tolerance_s)

                pt_sec, pt_imgs = get_processing_stats(processing_time_path)
                if pt_imgs > 0:
                    total_pt_seconds += pt_sec
                    total_pt_images += pt_imgs

                group_TP += res["TP"]
                group_FP += res["FP"]
                group_FN += res["FN"]
                group_total_gt += res["total_gt"]
                group_total_meas += res["total_meas"]
                group_acc_full += res["acc_full"]
                group_acc_action += res["acc_action"]
                group_acc_object += res["acc_object"]
                group_acc_on += res["acc_on"]
                group_acc_robot += res["acc_robot"]
                group_runs += 1

        group_precision = group_TP / (group_TP + group_FP) if (group_TP + group_FP) else 0.0
        group_recall = group_TP / (group_TP + group_FN) if (group_TP + group_FN) else 0.0
        group_f1 = (2 * group_precision * group_recall / (group_precision + group_recall)) if (group_precision + group_recall) else 0.0
        group_f1_list.append(group_f1)

        if group_runs:
            group_action_avg.append(group_acc_action / group_runs)
            group_object_avg.append(group_acc_object / group_runs)
            group_spatial_avg.append(group_acc_on / group_runs)
            group_robot_avg.append(group_acc_robot / group_runs)
            print("Accuracies:")
            print("  Full:   {:.2f}".format(group_acc_full / group_runs))
            print("  Action: {:.2f}".format(group_acc_action / group_runs))
            print("  Object: {:.2f}".format(group_acc_object / group_runs))
            print("  Spatial:{:.2f}".format(group_acc_on / group_runs))
            print("  Robot:  {:.2f}".format(group_acc_robot / group_runs))

        global_TP += group_TP
        global_FP += group_FP
        global_FN += group_FN
        global_total_gt += group_total_gt
        global_total_meas += group_total_meas
        global_acc_full += group_acc_full
        global_acc_action += group_acc_action
        global_acc_object += group_acc_object
        global_acc_on += group_acc_on
        global_acc_robot += group_acc_robot
        global_runs += group_runs

    global_precision = global_TP / (global_TP + global_FP) if (global_TP + global_FP) else 0.0
    global_recall = global_TP / (global_TP + global_FN) if (global_TP + global_FN) else 0.0
    global_f1 = (2 * global_precision * global_recall / (global_precision + global_recall)) if (global_precision + global_recall) else 0.0
    avg_runtime_per_image = (total_pt_seconds / total_pt_images) if total_pt_images > 0 else 0.0

    print("\n" + "="*50)
    print("GLOBAL MICRO TOTALS")
    print("Total GT:", global_total_gt)
    print("Total Meas:", global_total_meas)
    print("TP / FP / FN: {} / {} / {}".format(global_TP, global_FP, global_FN))
    print("Precision / Recall / F1: {:.2f} / {:.2f} / {:.2f}".format(global_precision, global_recall, global_f1))
    if global_runs:
        print("Accuracies:")
        print("  Full:   {:.2f}".format(global_acc_full / global_runs))
        print("  Action: {:.2f}".format(global_acc_action / global_runs))
        print("  Object: {:.2f}".format(global_acc_object / global_runs))
        print("  Spatial:{:.2f}".format(global_acc_on / global_runs))
        print("  Robot:  {:.2f}".format(global_acc_robot / global_runs))
    if total_pt_images > 0:
        print("Avg Processing Time (per image): {:.2f}s".format(avg_runtime_per_image))
    else:
        print("Avg Processing Time (per image): n/a")

    latex_row_f1 = " & ".join(f"{v:.2f}" for v in (group_f1_list + [global_f1, avg_runtime_per_image]))
    print(latex_row_f1)

    interleaved = []
    for i in range(len(group_action_avg)):
        interleaved.extend([group_action_avg[i], group_object_avg[i], group_spatial_avg[i], group_robot_avg[i]])

    overall_action = (global_acc_action / global_runs) if global_runs else 0.0
    overall_object = (global_acc_object / global_runs) if global_runs else 0.0
    overall_spatial = (global_acc_on / global_runs) if global_runs else 0.0
    overall_robot = (global_acc_robot / global_runs) if global_runs else 0.0

    latex_row_aosr = " & ".join(
        f"{v:.2f}" for v in (interleaved + [overall_action, overall_object, overall_spatial, overall_robot])
    )
    print(latex_row_aosr)
