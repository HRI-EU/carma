import os
import json

action_synonyms = {
    "hold": ["pick_up", "grasp"],
    "place_down": ["place"],
    "idle": ["idle"],
    "grasp": ["pick_up", "hold"],
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

def compute_accuracies(gt, meas, tolerance_s=1.0, print_candidates=True):
    ids = sorted(set(gt.keys()) & set(meas.keys()))
    total_gt = 0
    correct_full = 0
    correct_action = 0
    correct_object = 0
    correct_on = 0

    for pid in ids:
        gt_items = sorted(((float(ts), evt) for ts, evt in gt[pid].items()), key=lambda x: x[0])
        meas_items = sorted(((float(ts), evt) for ts, evt in meas[pid].items()), key=lambda x: x[0])
        used = [False] * len(meas_items)

        if print_candidates:
            print(f"\nID {pid}:")

        for tg, evg in gt_items:
            total_gt += 1
            key_g = event_key(evg)
            act_g, obj_g, on_g, _ = split_fields(evg)

            # All candidates within the time window
            candidates = [i for i, (tm, em) in enumerate(meas_items)
                          if not used[i] and abs(tm - tg) <= tolerance_s]

            if print_candidates and candidates:
                print(f" Ground Truth @ {tg}: {evg}")
                for i in candidates:
                    print(f"   Candidate @ {meas_items[i][0]}: {meas_items[i][1]}")

            # Check partial matches (action, object, on)
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

            # Full match
            full_candidates = [i for i in candidates if event_key(meas_items[i][1]) == key_g]
            if full_candidates:
                best_i = min(full_candidates, key=lambda i: abs(meas_items[i][0] - tg))
                used[best_i] = True
                correct_full += 1

    acc_full   = (correct_full   / total_gt) if total_gt else 0.0
    acc_action = (correct_action / total_gt) if total_gt else 0.0
    acc_object = (correct_object / total_gt) if total_gt else 0.0
    acc_on     = (correct_on     / total_gt) if total_gt else 0.0
    return acc_full, acc_action, acc_object, acc_on

if __name__ == "__main__":
    gt_path = "combined_data.json"
    meas_folder = "data/scene_009_PsortO/runs/True-True-True/"
    tolerance_s = 1

    ground_truth = strip_idle_from_gt(load_ground_truth(gt_path))
    measurements = load_measurements(meas_folder)

    acc_full, acc_action, acc_object, acc_on = compute_accuracies(
        ground_truth, measurements, tolerance_s=tolerance_s, print_candidates=True
    )

    print("\nResults (±{:.2f}s):".format(tolerance_s))
    print("  Accuracy (full event): {:.4f}".format(acc_full))
    print("  Accuracy (only action): {:.4f}".format(acc_action))
    print("  Accuracy (only object): {:.4f}".format(acc_object))
    print("  Accuracy (only on): {:.4f}".format(acc_on))
