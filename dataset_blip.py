from datasets import load_dataset
from transformers import BlipProcessor
from PIL import Image
import os

def preprocess_blip(example, image_dir, processor):
    image_path = os.path.join(image_dir, f"COCO_train2014_{str(example['image_id']).zfill(12)}.jpg")
    image = Image.open(image_path).convert("RGB")

    encoding = processor(
        images=image,
        text=example["caption"],
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )

    # 'input_ids' will be used both as input and as labels
    return {
        "pixel_values": encoding["pixel_values"].squeeze(0),
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
    }

def load_blip_dataset(annotation_file, image_dir):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    dataset = load_dataset("json", data_files=annotation_file, field="annotations")["train"]

    dataset = dataset.map(
        lambda e: preprocess_blip(e, image_dir, processor),
        remove_columns=dataset.column_names,
        batched=False
    )

    dataset.set_format(
        type="torch",
        columns=["pixel_values", "input_ids", "attention_mask"]
    )

    return dataset, processor
