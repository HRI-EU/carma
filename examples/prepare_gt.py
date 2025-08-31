import os
import json
from collections import defaultdict

runs = ["scene_009_PsortO", "scene_020_sf2P", "scene_021_sf2P", "scene_022_sf2P", "scene_026_sf1P1R", "scene_027_sf1P1R", "scene_029_sf2P1R", "scene_0290_sf2P1R",
        "scene_030_po2P", "scene_032_po2P", "scene_033_po1P1R", "scene_034_po1P1R", "scene_035_po1P1R", "scene_041_ha2P", "scene_042_ha2P", "scene_043_ha1P1R", 
        "scene_044_ha1P1R"]

for run in runs:
    folder = f"data/{run}/ground_truth/"
    output_file = f"data/{run}/ground_truth.json"

    # Rohdaten: {id: {timestamp_str: json_obj}}
    raw_dict = defaultdict(dict)

    # Dateien einlesen
    for filename in os.listdir(folder):
        if not filename.endswith(".json") or "_id_" not in filename:
            continue

        ts_str, rest = filename.split("_id_", 1)
        ts_str = ts_str.strip()
        file_id = rest.replace(".json", "").strip()

        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            json_obj = json.load(f)  # gesamte Datei (z. B. Liste von Dicts)

        raw_dict[file_id][ts_str] = json_obj

    # Übergänge extrahieren: nur erster Eintrag eines Zustandsblocks
    compressed = {}

    for file_id, ts_map in raw_dict.items():
        # Timestamps numerisch sortieren
        sorted_items = sorted(ts_map.items(), key=lambda kv: float(kv[0]))
        if not sorted_items:
            continue

        kept = {}
        prev_obj = None

        for ts_str, obj in sorted_items:
            if prev_obj is None or obj != prev_obj:  # voller Deep-Compare der Struktur
                kept[ts_str] = obj
                prev_obj = obj

        compressed[file_id] = kept

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(compressed, f, indent=4, ensure_ascii=False)

    print(f"✅ Übergänge gespeichert in {output_file}")
