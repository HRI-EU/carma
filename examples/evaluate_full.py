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

def get_processing_time(processing_time_path):
    if not os.path.isfile(processing_time_path):
        return 0.0
    with open(processing_time_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    pt = data.get('processing_time')
    images = data.get('images')
    return (pt / images) if images else 0.0

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

def normalize_action(action):
    for k, synonyms in action_synonyms.items():
        if action == k or action in synonyms:
            return k
    return action

def event_key(evt_list):
    if not isinstance(evt_list, list) or not evt_list or not isinstance(evt_list[0], dict):
        return ("", "", "", False)
    d = evt_list[0]
    act = normalize_action(d.get("action", ""))
    return (act, d.get("object", ""), d.get("on", ""), bool(d.get("robot_interaction", False)))

def split_fields(evt_list):
    if not isinstance(evt_list, list) or not evt_list or not isinstance(evt_list[0], dict):
        return ("", "", "", False)
    d = evt_list[0]
    return (
        normalize_action(d.get("action", "")),
        d.get("object", ""),
        d.get("on", ""),
        bool(d.get("robot_interaction", False)),
    )

def evaluate_run(gt, meas, tolerance_s=1.0, print_candidates=False):
    ids = sorted(set(gt.keys()) | set(meas.keys()))
    total_gt = 0
    correct_full = 0
    correct_action = 0
    correct_object = 0
    correct_on = 0
    total_meas = sum(len(meas.get(pid, {})) for pid in ids)
    used_global = 0

    for pid in ids:
        gt_items = sorted(((float(ts), evt) for ts, evt in gt.get(pid, {}).items()), key=lambda x: x[0])
        meas_items = sorted(((float(ts), evt) for ts, evt in meas.get(pid, {}).items()), key=lambda x: x[0])
        used = [False] * len(meas_items)

        if print_candidates and gt_items:
            print(f"\nID {pid}:")

        for tg, evg in gt_items:
            total_gt += 1
            key_g = event_key(evg)
            act_g, obj_g, on_g, _ = split_fields(evg)

            candidates = [i for i, (tm, em) in enumerate(meas_items)
                          if not used[i] and abs(tm - tg) <= tolerance_s]

            if print_candidates and candidates:
                print(50*"=")
                print(f" Ground Truth @ {tg}: {evg}")
                for i in candidates:
                    print(f"   Candidate @ {meas_items[i][0]}: {meas_items[i][1]}")

            matched_action = matched_object = matched_on = False
            for i in candidates:
                a_m, o_m, on_m, _ = split_fields(meas_items[i][1])
                if a_m == act_g:
                    matched_action = True
                if o_m == obj_g:
                    matched_object = True
                if on_m == on_g:
                    matched_on = True
                if matched_action and matched_object and matched_on:
                    break

            if matched_action:
                correct_action += 1
            if matched_object:
                correct_object += 1
            if matched_on:
                correct_on += 1

            full_candidates = [i for i in candidates if event_key(meas_items[i][1]) == key_g]
            if full_candidates:
                best_i = min(full_candidates, key=lambda i: abs(meas_items[i][0] - tg))
                used[best_i] = True
                used_global += 1
                correct_full += 1

    acc_full   = (correct_full   / total_gt) if total_gt else 0.0
    acc_action = (correct_action / total_gt) if total_gt else 0.0
    acc_object = (correct_object / total_gt) if total_gt else 0.0
    acc_on     = (correct_on     / total_gt) if total_gt else 0.0

    TP = correct_full
    FP = max(total_meas - used_global, 0)
    FN = max(total_gt - TP, 0)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "acc_full": acc_full,
        "acc_action": acc_action,
        "acc_object": acc_object,
        "acc_on": acc_on,
        "TP": TP, "FP": FP, "FN": FN,
        "precision": precision, "recall": recall, "f1": f1,
        "total_gt": total_gt, "total_meas": total_meas
    }

if __name__ == "__main__":
    experiments = [
        "scene_009_PsortO", "scene_020_sf2P", "scene_021_sf2P", "scene_022_sf2P",
        "scene_026_sf1P1R", "scene_027_sf1P1R", "scene_029_sf2P1R", "scene_0290_sf2P1R",
        "scene_030_po2P", "scene_032_po2P", "scene_033_po1P1R", "scene_034_po1P1R",
        "scene_041_ha2P", "scene_042_ha2P", "scene_043_ha1P1R", "scene_044_ha1P1R"
    ]

    model = "trigger-label-gpt-4o-0"
    model = "trigger-label-gemini-2.5-flash-0"
    tolerance_s = 5.0

    micro_TP = 0
    micro_FP = 0
    micro_FN = 0
    micro_total_gt = 0
    micro_total_meas = 0
    total_processing_time = 0.0
    run_count = 0

    for experiment in experiments:
        folder_pattern = f"data/{experiment}/runs/{model}"
        meas_folders = glob.glob(folder_pattern + "*")
        for meas_folder in meas_folders:
            run_count += 1
            print(f"\n---------------------------------- {meas_folder} ----------------------------------")
            gt_path = f"data/{experiment}/ground_truth.json"
            processing_time_path = f"{meas_folder}/processing_time.json"

            ground_truth = strip_idle_from_gt(load_ground_truth(gt_path))
            measurements = load_measurements(meas_folder)
            processing_time = get_processing_time(processing_time_path)

            res = evaluate_run(
                ground_truth, measurements, tolerance_s=tolerance_s, print_candidates=False
            )

            micro_TP += res["TP"]
            micro_FP += res["FP"]
            micro_FN += res["FN"]
            micro_total_gt += res["total_gt"]
            micro_total_meas += res["total_meas"]
            total_processing_time += processing_time

            print("Accuracy (full): {:.2f}".format(res["acc_full"]))
            print("Accuracy (action): {:.2f}".format(res["acc_action"]))
            print("Accuracy (object): {:.2f}".format(res["acc_object"]))
            print("Accuracy (on): {:.2f}".format(res["acc_on"]))
            print("TP / FP / FN: {} / {} / {}".format(res["TP"], res["FP"], res["FN"]))
            print("Precision / Recall / F1: {:.2f} / {:.2f} / {:.2f}".format(
                res["precision"], res["recall"], res["f1"]
            ))
            print("Processing Time (avg): {:.2f}s".format(processing_time))

    print("\n" + 50*"=")
    micro_precision = micro_TP / (micro_TP + micro_FP) if (micro_TP + micro_FP) else 0.0
    micro_recall = micro_TP / (micro_TP + micro_FN) if (micro_TP + micro_FN) else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) else 0.0
    avg_processing_time = (total_processing_time / run_count) if run_count else 0.0

    print("MICRO TOTALS")
    print("Total GT: {}".format(micro_total_gt))
    print("Total Meas: {}".format(micro_total_meas))
    print("TP / FP / FN: {} / {} / {}".format(micro_TP, micro_FP, micro_FN))
    print("Precision / Recall / F1: {:.2f} / {:.2f} / {:.2f}".format(micro_precision, micro_recall, micro_f1))
    print("Processing Time (avg): {:.2f}s".format(avg_processing_time))
