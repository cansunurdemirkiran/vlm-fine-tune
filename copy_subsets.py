import os
import json
import shutil
from tqdm import tqdm

root_dir = "data/coco_2015"
annotations_dir = os.path.join(root_dir, "annotations")
images_dir = os.path.join(root_dir, "train2014")
subsets = ["0.05", "0.01", "0.1"]

for subset in subsets:
    ann_path = os.path.join(annotations_dir, f"captions_train2014_{subset}.json")
    out_dir = os.path.join(root_dir, f"subsets/train2014_{subset}")
    os.makedirs(out_dir, exist_ok=True)

    # === Load annotation file ===
    with open(ann_path, "r") as f:
        data = json.load(f)

    # === Collect unique image_ids ===
    image_ids = set(ann["image_id"] for ann in data["annotations"])
    print(f"[{subset}%] Copying {len(image_ids)} images to {out_dir}...")

    # === Copy relevant images ===
    for img_id in tqdm(image_ids):
        filename = f"COCO_train2014_{str(img_id).zfill(12)}.jpg"
        src_path = os.path.join(images_dir, filename)
        dst_path = os.path.join(out_dir, filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"⚠️  Image not found: {filename}")

print("✅ Subset image folders created successfully.")