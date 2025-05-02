import os
import json
import shutil
from tqdm import tqdm

# Class mapping
DETECTION_CLASSES = [
    "bike",
    "bus",
    "car",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck",
]
CLASS_MAP = {name: i for i, name in enumerate(DETECTION_CLASSES)}


# YOLO conversion
def convert_annotation(json_path, label_path, img_w, img_h):
    with open(json_path, "r") as f:
        data = json.load(f)

    lines = []
    for obj in data["objects"]:
        if obj["geometryType"] != "rectangle":
            continue

        cls = obj["classTitle"]
        if cls not in CLASS_MAP:
            continue

        x1, y1 = obj["points"]["exterior"][0]
        x2, y2 = obj["points"]["exterior"][1]

        # YOLO format (normalized)
        xc = ((x1 + x2) / 2) / img_w
        yc = ((y1 + y2) / 2) / img_h
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h

        line = f"{CLASS_MAP[cls]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        lines.append(line)

    if lines:
        with open(label_path, "w") as out_f:
            out_f.write("\n".join(lines))


# Input/output config
splits = ["train", "val"]
for split in splits:
    ann_dir = f"dataset-ninja/{split}/ann"
    img_dir = f"dataset-ninja/{split}/img"
    out_img_dir = f"datasets/traffic/images/{split}"
    out_lbl_dir = f"datasets/traffic/labels/{split}"
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for filename in tqdm(os.listdir(ann_dir)):
        if not filename.endswith(".json"):
            continue

        base = filename.replace(".json", "").replace(".jpg", "")
        json_path = os.path.join(ann_dir, filename)
        img_path = os.path.join(img_dir, base + ".jpg")
        label_path = os.path.join(out_lbl_dir, base + ".txt")

        # Load image size from JSON
        with open(json_path, "r") as f:
            j = json.load(f)
            img_w, img_h = j["size"]["width"], j["size"]["height"]

        # Convert + copy image
        convert_annotation(json_path, label_path, img_w, img_h)
        shutil.copy(img_path, os.path.join(out_img_dir, base + ".jpg"))
