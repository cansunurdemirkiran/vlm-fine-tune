import os
import torch
import logging
from torch.utils.data import DataLoader
from transformers import (
    BlipForConditionalGeneration,
    Blip2ForConditionalGeneration,
    ViltForQuestionAnswering,
    AutoProcessor,
    GitForCausalLM,
    AutoTokenizer,
    AutoProcessor as GitProcessor
)

from dataset_blip import load_blip_dataset
from dataset_vilt import load_vilt_dataset
from dataset_git import load_git_dataset
from eval_utils import evaluate_model, log_metric, save_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_dir = "data/coco_2015/subsets"
annotations_dir = "data/coco_2015/annotations"
subsets = ["0.01", "0.05", "0.1"]
results = []

def log_progress(model_name, subset, epoch, batch_idx, total_batches, loss):
    logger.info(f"[{model_name}] Subset: {subset} | Epoch {epoch+1} | Batch {batch_idx+1}/{total_batches} | Loss: {loss:.4f}")

def log_start(model_name, subset):
    logger.info(f"\n=== {model_name.upper()} | Subset {subset} | Training Start ===")

def log_eval(model_name, subset):
    logger.info(f"🔍 Evaluating {model_name.upper()} on subset {subset}...")

def save_model(model, processor, name):
    out_dir = f"outputs/checkpoints/{name}"
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"📦 Saving model to {out_dir}")
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)

def train_blip():
    for subset in subsets:
        log_start("BLIP", subset)
        ann_file = f"{annotations_dir}/captions_train2014_{subset}.json"
        image_path = f"{image_dir}/train2014_{subset}"
        dataset, processor = load_blip_dataset(ann_file, image_path)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", use_safetensors=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        model.train()
        for epoch in range(3):
            logger.info(f"Epoch {epoch+1}/3 — Training BLIP...")
            for i, batch in enumerate(dataloader):
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    input_ids=batch["input_ids"].to(device),
                    labels=batch["input_ids"].to(device)
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                log_progress("BLIP", subset, epoch, i, len(dataloader), loss.item())

        log_eval("BLIP", subset)
        metrics = evaluate_model(model, processor, ann_file, image_path, device)
        log_metric(results, "BLIP", subset, metrics)
        save_model(model, processor, f"blip_{subset}")

def train_git():
    for subset in subsets:
        log_start("GIT", subset)
        ann_file = f"{annotations_dir}/captions_train2014_{subset}.json"
        image_path = f"{image_dir}/train2014_{subset}"
        dataset, processor = load_git_dataset(ann_file, image_path)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        model = GitForCausalLM.from_pretrained("microsoft/git-base").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        model.train()
        for epoch in range(3):
            logger.info(f"Epoch {epoch+1}/3 — Training GIT...")
            for i, batch in enumerate(dataloader):
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    input_ids=batch["input_ids"].to(device),
                    labels=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device)
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                log_progress("GIT", subset, epoch, i, len(dataloader), loss.item())

        log_eval("GIT", subset)
        metrics = evaluate_model(model, processor, ann_file, image_path, device)
        log_metric(results, "GIT", subset, metrics)
        save_model(model, processor, f"git_{subset}")

def train_vilt():
    for subset in subsets:
        log_start("ViLT", subset)
        ann_file = f"{annotations_dir}/captions_train2014_{subset}.json"
        image_path = f"{image_dir}/train2014_{subset}"
        dataset, processor = load_vilt_dataset(ann_file, image_path)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        model.train()
        for epoch in range(3):
            logger.info(f"Epoch {epoch+1}/3 — Training ViLT...")
            for i, batch in enumerate(dataloader):
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    input_ids=batch["input_ids"].to(device),
                    labels=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device)
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                log_progress("ViLT", subset, epoch, i, len(dataloader), loss.item())

        log_eval("ViLT", subset)
        metrics = evaluate_model(model, processor, ann_file, image_path, device)
        log_metric(results, "ViLT", subset, metrics)
        save_model(model, processor, f"vilt_{subset}")

if __name__ == "__main__":
    logger.info("🚀 Starting training for all models...")
    train_vilt()
    train_blip()
    train_git()
    save_metrics()
    logger.info("✅ Training and evaluation complete.")