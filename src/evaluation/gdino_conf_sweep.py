"""
Grounding DINO confidence-threshold sensitivity sweep (Tiny, class-name prompt),
at IoU thresholds 0.10/0.25/0.50, to test whether the box/text confidence
threshold (fixed at 0.25/0.20 in the main ablation) hides non-zero performance
at lower confidence.

Run: python src/evaluation/gdino_conf_sweep.py
"""
import json
import warnings
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

from ablation_gdino import compute_ap, match_detections, yolo_label_to_boxes

ROOT = Path(__file__).parents[2]
TEST_DIR = ROOT / "data" / "dentex_yolo" / "test"
OUT = ROOT / "results" / "evaluation" / "gdino_confidence_sweep.json"

CLASS_NAMES = ["Caries", "Periapical lesion", "Impacted tooth"]
PROMPT = ["caries", "periapical lesion", "impacted tooth"]
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
IOU_SWEEP = [0.10, 0.25, 0.50]
CONF_THRESHOLDS = [0.05, 0.10, 0.15, 0.25]
TEXT_THRESH = 0.20


def run_inference_min_conf(model_id, prompt, box_thresh):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    gdino.eval()

    text_labels = [prompt]
    raw = {}
    img_files = sorted((TEST_DIR / "images").glob("*.jpg"))
    for img_f in tqdm(img_files, desc=f"conf>={box_thresh}"):
        lbl_f = TEST_DIR / "labels" / (img_f.stem + ".txt")
        img = Image.open(img_f).convert("RGB")
        w, h = img.size
        gts = yolo_label_to_boxes(lbl_f, w, h)

        inputs = processor(images=img, text=text_labels, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = gdino(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs, threshold=box_thresh, text_threshold=TEXT_THRESH,
            target_sizes=[(h, w)], text_labels=text_labels,
        )[0]

        preds = []
        for score, raw_lbl, box in zip(results["scores"], results["text_labels"], results["boxes"]):
            raw_l = str(raw_lbl).lower().strip()
            cls = None
            for i, name in enumerate(prompt):
                if name.lower() in raw_l or raw_l in name.lower():
                    cls = i
                    break
            if cls is None:
                continue
            x1, y1, x2, y2 = box.tolist()
            preds.append((cls, float(score), x1, y1, x2, y2))

        raw[img_f.stem] = {"gts": gts, "preds": preds}

    del gdino, processor
    return raw


def score_config(raw, iou_thresh):
    per_class_tp_fp = defaultdict(list)
    per_class_n_gt = defaultdict(int)
    for entry in raw.values():
        for cls, *_ in entry["gts"]:
            per_class_n_gt[cls] += 1
        matches = match_detections(entry["preds"], entry["gts"], iou_thresh)
        for is_tp, score, cls in matches:
            per_class_tp_fp[cls].append((is_tp, score))

    aps = {}
    for cls in range(len(CLASS_NAMES)):
        ap = compute_ap(per_class_tp_fp.get(cls, []), per_class_n_gt.get(cls, 0))
        aps[CLASS_NAMES[cls]] = round(ap, 4)
    mAP = round(sum(aps.values()) / len(aps), 4)
    return {"mAP": mAP, "per_class_ap": aps}


def main():
    results = {}
    for box_thresh in CONF_THRESHOLDS:
        raw = run_inference_min_conf(MODEL_ID, PROMPT, box_thresh)
        results[f"conf_{box_thresh}"] = {
            f"iou_{iou}": score_config(raw, iou) for iou in IOU_SWEEP
        }
        del raw

    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
