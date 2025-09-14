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
    total_meas = sum(len(meas.get(pid, {})) for pid in ids)

    tp_full = 0
    tp_a = tp_o = tp_on = tp_r = 0

    correct_action = correct_object = correct_on = correct_robot = 0  # kept only if you still want per-field "accuracy" later

    for pid in ids:
        gt_items = sorted(((float(ts), evt) for ts, evt in gt.get(pid, {}).items()), key=lambda x: x[0])
        meas_items = sorted(((float(ts), evt) for ts, evt in meas.get(pid, {}).items()), key=lambda x: x[0])

        used_idx_full = [False] * len(meas_items)
        used_idx_a    = [False] * len(meas_items)
        used_idx_o    = [False] * len(meas_items)
        used_idx_on   = [False] * len(meas_items)
        used_idx_r    = [False] * len(meas_items)

        for tg, evg in gt_items:
            total_gt += 1
            ag, og, ong, rg = event_key(evg)

            cand = [i for i,(tm,_) in enumerate(meas_items) if abs(tm - tg) <= tolerance_s]

            hit_a = hit_o = hit_on = hit_r = False
            best_full_i = None
            best_a_i = best_o_i = best_on_i = best_r_i = None
            best_full_dt = best_a_dt = best_o_dt = best_on_dt = best_r_dt = float("inf")

            for i in cand:
                am, om, onm, rm = event_key(meas_items[i][1])
                dt = abs(meas_items[i][0] - tg)

                if am == ag and not used_idx_a[i] and dt < best_a_dt:
                    best_a_dt, best_a_i = dt, i
                if om == og and not used_idx_o[i] and dt < best_o_dt:
                    best_o_dt, best_o_i = dt, i
                if onm == ong and not used_idx_on[i] and dt < best_on_dt:
                    best_on_dt, best_on_i = dt, i
                if rm == rg and not used_idx_r[i] and dt < best_r_dt:
                    best_r_dt, best_r_i = dt, i
                if am == ag and om == og and onm == ong and rm == rg and not used_idx_full[i] and dt < best_full_dt:
                    best_full_dt, best_full_i = dt, i

                if am == ag:  hit_a = True
                if om == og:  hit_o = True
                if onm == ong: hit_on = True
                if rm == rg:  hit_r = True

            if hit_a:  correct_action += 1
            if hit_o:  correct_object += 1
            if hit_on: correct_on     += 1
            if hit_r:  correct_robot  += 1

            if best_a_i is not None:
                used_idx_a[best_a_i] = True
                tp_a += 1
            if best_o_i is not None:
                used_idx_o[best_o_i] = True
                tp_o += 1
            if best_on_i is not None:
                used_idx_on[best_on_i] = True
                tp_on += 1
            if best_r_i is not None:
                used_idx_r[best_r_i] = True
                tp_r += 1
            if best_full_i is not None:
                used_idx_full[best_full_i] = True
                tp_full += 1

    fp_full = total_meas - tp_full
    fn_full = total_gt - tp_full

    fp_a  = total_meas - tp_a
    fp_o  = total_meas - tp_o
    fp_on = total_meas - tp_on
    fp_r  = total_meas - tp_r

    fn_a  = total_gt - tp_a
    fn_o  = total_gt - tp_o
    fn_on = total_gt - tp_on
    fn_r  = total_gt - tp_r

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2*p*r/(p+r) if (p+r) else 0.0
        return p, r, f

    p_full, r_full, f_full = prf(tp_full, fp_full, fn_full)
    _, _, f_a  = prf(tp_a,  fp_a,  fn_a)
    _, _, f_o  = prf(tp_o,  fp_o,  fn_o)
    _, _, f_on = prf(tp_on, fp_on, fn_on)
    _, _, f_r  = prf(tp_r,  fp_r,  fn_r)

    acc_full   = tp_full / total_gt if total_gt else 0.0
    acc_action = correct_action / total_gt if total_gt else 0.0
    acc_object = correct_object / total_gt if total_gt else 0.0
    acc_on     = correct_on / total_gt if total_gt else 0.0
    acc_robot  = correct_robot / total_gt if total_gt else 0.0

    return {
        "TP": tp_full, "FP": fp_full, "FN": fn_full,
        "precision": p_full, "recall": r_full, "f1": f_full,
        "total_gt": total_gt, "total_meas": total_meas,
        "acc_full": acc_full, "acc_action": acc_action, "acc_object": acc_object, "acc_on": acc_on, "acc_robot": acc_robot,
        "TP_a": tp_a, "FP_a": fp_a, "FN_a": fn_a, "f1_a": f_a,
        "TP_o": tp_o, "FP_o": fp_o, "FN_o": fn_o, "f1_o": f_o,
        "TP_on": tp_on, "FP_on": fp_on, "FN_on": fn_on, "f1_on": f_on,
        "TP_r": tp_r, "FP_r": fp_r, "FN_r": fn_r, "f1_r": f_r,
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

    experiments_groups = [
        ["scene_009_PsortO"],
        ["scene_020_sf2P", "scene_021_sf2P", "scene_022_sf2P"],
        ["scene_026_sf1P1R", "scene_027_sf1P1R"],
        ["scene_029_sf2P1R", "scene_0290_sf2P1R"],
        ["scene_030_po2P", "scene_032_po2P"],
        ["scene_033_po1P1R", "scene_034_po1P1R"],
        ["scene_041_ha2P", "scene_042_ha2P"],
        ["scene_043_ha1P1R", "scene_044_ha1P1R"]
    ]

    model = "gpt-5-0"    
    # model = "trigger-label-full-gpt-5-0"
    # model = "trigger-label-gpt-5-0"
    # model = "gemini-2.5-flash-0"
    model = "trigger-label-full-gemini-2.5-flash-0"
    # model = "trigger-label-gemini-2.5-flash-0"    
    # model = "gpt-4o-0"
    model = "trigger-label-full-gpt-4o-0"
    # model = "trigger-label-gpt-4o-0"
    
    tolerance_s = 5.0

    global_TP = global_FP = global_FN = 0
    global_TP_a = global_FP_a = global_FN_a = 0
    global_TP_o = global_FP_o = global_FN_o = 0
    global_TP_on = global_FP_on = global_FN_on = 0
    global_TP_r = global_FP_r = global_FN_r = 0

    total_pt_seconds = 0.0
    total_pt_images = 0

    group_f1_list = []
    group_f1_action = []
    group_f1_object = []
    group_f1_spatial = []
    group_f1_robot = []

    for gi, group in enumerate(experiments_groups, start=1):
        gTP = gFP = gFN = 0
        gTP_a = gFP_a = gFN_a = 0
        gTP_o = gFP_o = gFN_o = 0
        gTP_on = gFP_on = gFN_on = 0
        gTP_r = gFP_r = gFN_r = 0

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
                    if "trigger" in model:
                        ocad_pt = pt_imgs * 0.2
                    else:
                        ocad_pt = 0.0
                    total_pt_seconds += (pt_sec + ocad_pt)
                    total_pt_images += pt_imgs

                gTP += res["TP"];     gFP += res["FP"];     gFN += res["FN"]
                gTP_a += res["TP_a"]; gFP_a += res["FP_a"]; gFN_a += res["FN_a"]
                gTP_o += res["TP_o"]; gFP_o += res["FP_o"]; gFN_o += res["FN_o"]
                gTP_on += res["TP_on"]; gFP_on += res["FP_on"]; gFN_on += res["FN_on"]
                gTP_r += res["TP_r"]; gFP_r += res["FP_r"]; gFN_r += res["FN_r"]

        def prf(tp, fp, fn):
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f = 2*p*r/(p+r) if (p+r) else 0.0
            return p, r, f

        _, _, gF_full = prf(gTP, gFP, gFN)
        _, _, gF_a    = prf(gTP_a, gFP_a, gFN_a)
        _, _, gF_o    = prf(gTP_o, gFP_o, gFN_o)
        _, _, gF_on   = prf(gTP_on, gFP_on, gFN_on)
        _, _, gF_r    = prf(gTP_r, gFP_r, gFN_r)

        group_f1_list.append(gF_full)
        group_f1_action.append(gF_a)
        group_f1_object.append(gF_o)
        group_f1_spatial.append(gF_on)
        group_f1_robot.append(gF_r)

        global_TP += gTP;       global_FP += gFP;       global_FN += gFN
        global_TP_a += gTP_a;   global_FP_a += gFP_a;   global_FN_a += gFN_a
        global_TP_o += gTP_o;   global_FP_o += gFP_o;   global_FN_o += gFN_o
        global_TP_on += gTP_on; global_FP_on += gFP_on; global_FN_on += gFN_on
        global_TP_r += gTP_r;   global_FP_r += gFP_r;   global_FN_r += gFN_r

        print("Precision/Recall/F1 (full): {:.2f} / {:.2f} / {:.2f}".format(*prf(gTP, gFP, gFN)))

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2*p*r/(p+r) if (p+r) else 0.0
        return p, r, f

    _, _, global_f1 = prf(global_TP, global_FP, global_FN)
    _, _, overall_f1_action  = prf(global_TP_a,  global_FP_a,  global_FN_a)
    _, _, overall_f1_object  = prf(global_TP_o,  global_FP_o,  global_FN_o)
    _, _, overall_f1_spatial = prf(global_TP_on, global_FP_on, global_FN_on)
    _, _, overall_f1_robot   = prf(global_TP_r,  global_FP_r,  global_FN_r)

    avg_runtime_per_image = (total_pt_seconds / total_pt_images) if total_pt_images > 0 else 0.0

    print("\n" + "="*50)
    print("GLOBAL MICRO TOTALS (F1 only shown below)")
    print("Full F1: {:.2f}".format(global_f1))
    print("Action/Object/Spatial/Robot F1: {:.2f} / {:.2f} / {:.2f} / {:.2f}".format(
        overall_f1_action, overall_f1_object, overall_f1_spatial, overall_f1_robot))
    if total_pt_images > 0:
        print("Avg Processing Time (per image): {:.2f}s".format(avg_runtime_per_image))
    else:
        print("Avg Processing Time (per image): n/a")

    latex_row_f1 = " & ".join(f"{v:.2f}" for v in (group_f1_list + [global_f1, avg_runtime_per_image]))
    print(latex_row_f1)

    interleaved = []
    for i in range(len(group_f1_action)):
        interleaved.extend([group_f1_action[i], group_f1_object[i], group_f1_spatial[i], group_f1_robot[i]])

    latex_row_fields = " & ".join(
        f"{v:.2f}" for v in (
            interleaved + [overall_f1_action, overall_f1_object, overall_f1_spatial, overall_f1_robot]
        )
    )
    print(latex_row_fields)
