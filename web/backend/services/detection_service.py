import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

YOLO_CLASS_TO_CATEGORY = {
    "Caries":            "Diseases",
    "Periapical lesion": "Diseases",
    "Impacted tooth":    "Tooth Status",
}


def run_stage1(model, image_path: Path, annotated_out: Path,
               model_type: str = "yolo", **kwargs) -> list[dict]:
    """
    Unified detection entry point.
    model_type="yolo"  → YOLOv11 fine-tuned (primary)
    model_type="gdino" → Grounding DINO zero-shot (fallback)
    """
    from services.image_service import render_annotated

    if model_type == "yolo":
        det_dicts = _run_yolo(model, image_path)
    else:
        processor = kwargs.get("processor")
        device    = kwargs.get("device", "cpu")
        det_dicts = _run_gdino(processor, model, device, image_path)

    render_annotated(image_path, det_dicts, [], annotated_out)
    return det_dicts


def _run_yolo(model, image_path: Path) -> list[dict]:
    results = model(str(image_path), conf=0.25, iou=0.45, verbose=False)[0]
    dets = []
    for box in results.boxes:
        cls_name = model.names[int(box.cls)]
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        dets.append({
            "box_x1": x1, "box_y1": y1, "box_x2": x2, "box_y2": y2,
            "class_name": cls_name,
            "raw_label":  cls_name.lower(),
            "score":      round(float(box.conf), 4),
            "category":   YOLO_CLASS_TO_CATEGORY.get(cls_name, "Diseases"),
            "is_valid":   1,
        })
    return dets


def _run_gdino(processor, model, device, image_path: Path) -> list[dict]:
    """Fallback — Grounding DINO zero-shot for when YOLO model is not trained yet."""
    sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "pipeline"))
    from stage1_detection import detect_image as gdino_detect

    result = gdino_detect(processor, model, device, image_path)
    return [
        {"box_x1": d["box"][0], "box_y1": d["box"][1],
         "box_x2": d["box"][2], "box_y2": d["box"][3],
         "class_name": d["class"], "raw_label": d.get("raw_label", ""),
         "score": d["score"], "category": d["category"], "is_valid": 1}
        for d in result["detections"]
    ]
