from evaluate import load
from PIL import Image

def evaluate_model(model, processor, image_caption_pairs, device="cuda"):
    model.eval()
    bleu = load("bleu")
    cider = load("cider")
    meteor = load("meteor")

    preds, refs = [], []
    for image_path, reference in image_caption_pairs:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        output = model.generate(**inputs)
        caption = processor.tokenizer.decode(output[0], skip_special_tokens=True)
        preds.append(caption)
        refs.append([reference])

    return {
        "BLEU": bleu.compute(predictions=preds, references=refs),
        "CIDEr": cider.compute(predictions=preds, references=refs),
        "METEOR": meteor.compute(predictions=preds, references=refs),
    }

import os
import pandas as pd

evaluation_results = []

def log_metric(model_name, subset, metrics_dict):
    evaluation_results.append({
        "Model": model_name,
        "Subset": f"{subset}%",
        "BLEU": round(metrics_dict["BLEU"]["bleu"], 4),
        "CIDEr": round(metrics_dict["CIDEr"]["cider"], 4),
        "METEOR": round(metrics_dict["METEOR"]["meteor"], 4),
    })

def save_metrics(path="outputs/metrics/evaluation_results.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(evaluation_results)
    df.to_csv(path, index=False)
    print(f"\n📄 Evaluation results saved to: {path}")
