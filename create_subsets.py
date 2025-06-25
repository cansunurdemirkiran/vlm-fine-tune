import json
import os
import random

base_json_path = "data/coco_2015/annotations/captions_train2014.json"
output_dir = "data/coco_2015/annotations"
subsets = {
    "0.01": 0.001,
    "0.05": 0.005,
    "0.1": 0.01
}
random.seed(42)

with open(base_json_path, 'r') as f:
    coco_data = json.load(f)

annotations = coco_data["annotations"]
info = coco_data.get("info", {})
licenses = coco_data.get("licenses", [])
images = coco_data.get("images", [])

print(f"Total annotations: {len(annotations)}")

for name, ratio in subsets.items():
    subset_size = int(len(annotations) * ratio)
    sampled_annotations = random.sample(annotations, subset_size)

    image_ids = {ann["image_id"] for ann in sampled_annotations}
    subset_images = [img for img in images if img["id"] in image_ids]

    subset_data = {
        "info": info,
        "licenses": licenses,
        "images": subset_images,
        "annotations": sampled_annotations
    }

    subset_path = os.path.join(output_dir, f"captions_train2014_{name}.json")
    with open(subset_path, 'w') as f:
        json.dump(subset_data, f)

    print(f"Saved {subset_path} with {len(sampled_annotations)} annotations and {len(subset_images)} images.")