"""
Stage 1 — Zero-shot dental pathology detection using Grounding DINO.
Processes all 50 OPGs and saves detection results + annotated images.
"""
import json
import sys
import warnings
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    IMAGES_DIR, RESULTS_DIR, GDINO_MODEL_ID,
    CLASS_NAMES, CLASS_TO_CATEGORY, CATEGORY_COLORS,
    BOX_THRESHOLD, TEXT_THRESHOLD,
)

# Prompt individual por classe — melhor matching no transformers 5.x
TEXT_LABELS = [[c.lower() for c in CLASS_NAMES]]


def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Carregando Grounding DINO em: {device}")
    processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to(device)
    model.eval()
    return processor, model, device


def match_label_to_class(raw_label: str) -> str:
    """Map a raw GDINO text span back to the closest CLASS_NAMES entry."""
    raw = raw_label.lower().strip()
    # exact match first
    for cls in CLASS_NAMES:
        if cls.lower() == raw:
            return cls
    # substring match (longest wins)
    best, best_len = "Unknown", 0
    for cls in CLASS_NAMES:
        cls_l = cls.lower()
        if cls_l in raw or raw in cls_l:
            if len(cls_l) > best_len:
                best, best_len = cls, len(cls_l)
    return best


def detect_image(processor, model, device, image_path: Path) -> dict:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=TEXT_LABELS, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
        text_labels=TEXT_LABELS,
    )[0]

    detections = []
    for score, raw_label, box in zip(
        results["scores"], results["text_labels"], results["boxes"]
    ):
        cls_name = match_label_to_class(str(raw_label))
        cls_idx = CLASS_NAMES.index(cls_name) if cls_name in CLASS_NAMES else -1
        detections.append({
            "class": cls_name,
            "raw_label": str(raw_label),
            "score": round(float(score), 4),
            "box": [round(float(v), 1) for v in box.tolist()],
            "category": CLASS_TO_CATEGORY.get(cls_idx, "Unknown"),
        })

    detections.sort(key=lambda d: d["score"], reverse=True)
    return {"image": image_path.name, "detections": detections, "count": len(detections)}


def draw_detections(image_path: Path, result: dict, out_path: Path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for det in result["detections"]:
        x1, y1, x2, y2 = det["box"]
        color = CATEGORY_COLORS.get(det["category"], (128, 128, 128))
        draw.rectangle([x1, y1, x2, y2], fill=color + (40,), outline=color + (220,), width=3)
        label = f"{det['class']} {det['score']:.2f}"
        draw.rectangle([x1, y1 - 20, x1 + len(label) * 8, y1], fill=color + (200,))
        draw.text((x1 + 2, y1 - 17), label, fill="white")
    image.save(out_path, quality=90)


def run():
    out_dir = RESULTS_DIR / "stage1_detections"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = RESULTS_DIR / "figures" / "stage1"
    vis_dir.mkdir(parents=True, exist_ok=True)

    processor, model, device = load_model()
    image_paths = sorted(IMAGES_DIR.glob("Image_*.jpg"))
    all_results = []

    for img_path in tqdm(image_paths, desc="Detectando"):
        result = detect_image(processor, model, device, img_path)
        all_results.append(result)

        img_id = img_path.stem.replace("Image_", "")
        (out_dir / f"detection_{img_id}.json").write_text(json.dumps(result, indent=2))
        draw_detections(img_path, result, vis_dir / f"det_{img_id}.jpg")

    total_dets = sum(r["count"] for r in all_results)
    class_freq: dict[str, int] = {}
    for r in all_results:
        for d in r["detections"]:
            class_freq[d["class"]] = class_freq.get(d["class"], 0) + 1

    summary = {
        "total_images": len(all_results),
        "total_detections": total_dets,
        "avg_detections_per_image": round(total_dets / len(all_results), 2),
        "classes_detected": len(class_freq),
        "class_frequency": dict(sorted(class_freq.items(), key=lambda x: -x[1])),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✓ Stage 1 | {total_dets} detecções | {len(class_freq)} classes | média {summary['avg_detections_per_image']}/imagem")
    return all_results


if __name__ == "__main__":
    run()
