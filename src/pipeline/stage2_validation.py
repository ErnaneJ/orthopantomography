"""
Stage 2 — Spontaneous Recall Evaluation (corrected, bias-free).

Previous version fed expected classes back to the detector as prompts (circular).
This version computes Spontaneous Recall Rate:
  SR = |YOLO detected classes| ∩ |dentist-mentioned classes| / |dentist-mentioned classes|

Restricted to the 3 classes YOLO was trained on:
  Caries, Periapical lesion, Impacted tooth

Only measures what the model detects spontaneously without any hint.
"""
import json
import sys
import numpy as np
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import IMAGES_DIR, RESULTS_DIR
from utils import load_description, match_findings_to_classes, extract_findings_from_text

# Classes YOLO was trained on — only these can be spontaneously recalled
YOLO_CLASSES = {"Caries", "Periapical lesion", "Impacted tooth"}


def compute_spontaneous_recall(img_id: str, s1_result: dict) -> dict:
    description = load_description(img_id)
    findings_text = extract_findings_from_text(description)
    all_mentioned = set(match_findings_to_classes(findings_text))

    # Restrict to classes YOLO was trained on
    testable_mentioned = all_mentioned & YOLO_CLASSES
    detected_classes   = {d["class"] for d in s1_result.get("detections", [])}

    if not testable_mentioned:
        return {
            "image": s1_result["image"],
            "description": description,
            "all_mentioned_classes": list(all_mentioned),
            "testable_classes": [],
            "detected_yolo_classes": list(detected_classes),
            "matched": [],
            "spontaneous_recall": None,
            "note": "No YOLO-trainable classes mentioned in description",
        }

    matched = testable_mentioned & detected_classes
    recall  = len(matched) / len(testable_mentioned)

    return {
        "image": s1_result["image"],
        "description": description,
        "all_mentioned_classes": list(all_mentioned),
        "testable_classes": list(testable_mentioned),
        "detected_yolo_classes": list(detected_classes),
        "matched": list(matched),
        "spontaneous_recall": round(recall, 4),
    }


def run(stage1_results: list[dict] | None = None):
    out_dir = RESULTS_DIR / "stage2_validations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load stage1 results from disk if not passed in
    s1_index = {}
    if stage1_results:
        s1_index = {r["image"]: r for r in stage1_results}
    else:
        s1_dir = RESULTS_DIR / "stage1_detections"
        for f in s1_dir.glob("detection_*.json"):
            d = json.loads(f.read_text())
            s1_index[d["image"]] = d

    image_paths = sorted(IMAGES_DIR.glob("Image_*.jpg"))
    all_results, recall_scores = [], []

    for img_path in tqdm(image_paths, desc="Spontaneous recall"):
        img_id   = img_path.stem.replace("Image_", "")
        img_name = img_path.name
        s1       = s1_index.get(img_name, {"image": img_name, "detections": []})

        result = compute_spontaneous_recall(img_id, s1)
        all_results.append(result)

        out_f = out_dir / f"validation_{img_id}.json"
        out_f.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        if result["spontaneous_recall"] is not None:
            recall_scores.append(result["spontaneous_recall"])

    avg = round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else 0.0
    summary = {
        "metric": "Spontaneous Recall Rate (unbiased — no prompt injection)",
        "yolo_classes_evaluated": list(YOLO_CLASSES),
        "total_images": len(all_results),
        "images_with_testable_findings": len(recall_scores),
        "avg_spontaneous_recall": avg,
        "std_spontaneous_recall": round(float(np.std(recall_scores)), 4) if recall_scores else 0.0,
        "recall_distribution": {
            "high_>=0.8": sum(1 for s in recall_scores if s >= 0.8),
            "mid_0.5-0.8": sum(1 for s in recall_scores if 0.5 <= s < 0.8),
            "low_<0.5": sum(1 for s in recall_scores if s < 0.5),
            "zero": sum(1 for s in recall_scores if s == 0.0),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n✓ Stage 2 (Spontaneous Recall) | Mean: {avg:.1%} | "
          f"Images evaluated: {len(recall_scores)}")
    return all_results


if __name__ == "__main__":
    run()
