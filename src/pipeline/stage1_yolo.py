"""
Stage 1 (v2) — YOLOv11 detection on OPG images.
Replaces Grounding DINO zero-shot with a model fine-tuned on DentexChallenge 2023.
Classes: Caries (0), Periapical lesion (1), Impacted tooth (2)

Usage (standalone):
  python src/pipeline/stage1_yolo.py
"""
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    IMAGES_DIR, RESULTS_DIR, YOLO_MODEL_PATH,
    YOLO_CONF, YOLO_IOU, YOLO_CLASS_TO_CATEGORY,
)


def load_model():
    from ultralytics import YOLO
    if not YOLO_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model not found at {YOLO_MODEL_PATH}\n"
            "Train first: python src/training/train_yolo.py"
        )
    model = YOLO(str(YOLO_MODEL_PATH))
    print(f"✓ YOLO model loaded from {YOLO_MODEL_PATH.name}")
    return model


def detect_image(model, image_path: Path) -> dict:
    results = model(
        str(image_path),
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        verbose=False,
    )[0]

    detections = []
    for box in results.boxes:
        cls_id   = int(box.cls)
        cls_name = model.names[cls_id]
        score    = float(box.conf)
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        detections.append({
            "class":    cls_name,
            "class_name": cls_name,
            "raw_label": cls_name.lower(),
            "score":    round(score, 4),
            "box":      [x1, y1, x2, y2],
            "box_x1": x1, "box_y1": y1, "box_x2": x2, "box_y2": y2,
            "category": YOLO_CLASS_TO_CATEGORY.get(cls_name, "Diseases"),
            "is_valid": 1,
        })

    detections.sort(key=lambda d: d["score"], reverse=True)
    return {"image": image_path.name, "detections": detections, "count": len(detections)}


def draw_detections(image_path: Path, result: dict, out_path: Path):
    from PIL import ImageDraw
    from config import CATEGORY_COLORS
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    for det in result["detections"]:
        x1, y1, x2, y2 = det["box"]
        color = CATEGORY_COLORS.get(det["category"], (128, 128, 128))
        draw.rectangle([x1, y1, x2, y2], fill=color + (40,), outline=color + (220,), width=3)
        label = f"{det['class']} {det['score']:.2f}"
        draw.rectangle([x1, y1 - 20, x1 + len(label) * 8, y1], fill=color + (200,))
        draw.text((x1 + 2, y1 - 17), label, fill="white")
    img.save(out_path, quality=90)


def run():
    from tqdm import tqdm

    out_dir = RESULTS_DIR / "stage1_detections"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = RESULTS_DIR / "figures" / "stage1"
    vis_dir.mkdir(parents=True, exist_ok=True)

    model = load_model()
    image_paths = sorted(IMAGES_DIR.glob("Image_*.jpg"))
    all_results = []

    for img_path in tqdm(image_paths, desc="Detecting (YOLO)"):
        result = detect_image(model, img_path)
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
        "model": "YOLOv11 (fine-tuned on DentexChallenge 2023)",
        "total_images": len(all_results),
        "total_detections": total_dets,
        "avg_detections_per_image": round(total_dets / max(len(all_results), 1), 2),
        "classes_detected": len(class_freq),
        "class_frequency": dict(sorted(class_freq.items(), key=lambda x: -x[1])),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✓ Stage 1 (YOLO) | {total_dets} detections | {len(class_freq)} classes | "
          f"avg {summary['avg_detections_per_image']}/image")
    return all_results


if __name__ == "__main__":
    run()
