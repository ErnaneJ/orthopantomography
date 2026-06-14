"""
Compare YOLOv11 (fine-tuned) vs Grounding DINO (zero-shot) on the same
DentexChallenge 2023 test split. Generates the comparison table for the paper.

Run: python src/evaluation/compare_baselines.py
"""
import json
import warnings
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parents[2]
TEST_DIR = ROOT / "data" / "dentex_yolo" / "test"
OUT      = ROOT / "results" / "evaluation"

# Dentex class names for the 3-class model
CLASS_NAMES = ["Caries", "Periapical lesion", "Impacted tooth"]

# IOU threshold for TP/FP matching
IOU_THRESH = 0.5
CONF_THRESH = 0.25


def compute_iou(box_a, box_b):
    """boxes in [x1,y1,x2,y2] pixel format."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1); inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2); inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def yolo_label_to_boxes(label_path: Path, img_w: int, img_h: int) -> list:
    """Read YOLO .txt → list of (class_id, x1, y1, x2, y2)."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, nw, nh = int(parts[0]), *[float(x) for x in parts[1:5]]
        x1 = (cx - nw / 2) * img_w
        y1 = (cy - nh / 2) * img_h
        x2 = (cx + nw / 2) * img_w
        y2 = (cy + nh / 2) * img_h
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def match_detections(preds: list, gts: list, iou_thresh: float = IOU_THRESH):
    """
    preds: [(cls, score, x1, y1, x2, y2), ...]
    gts:   [(cls, x1, y1, x2, y2), ...]
    Returns: list of (is_tp, score, cls) for each pred
    """
    matched_gt = set()
    results = []
    # Sort by score descending
    for pred_cls, score, *pred_box in sorted(preds, key=lambda x: -x[1]):
        best_iou, best_gt_idx = 0.0, -1
        for i, (gt_cls, *gt_box) in enumerate(gts):
            if i in matched_gt or gt_cls != pred_cls:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_gt_idx = iou, i
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            results.append((True, score, pred_cls))
        else:
            results.append((False, score, pred_cls))
    return results, len(gts)


def compute_ap(tp_fp_list: list, n_gt: int) -> float:
    """11-point AP interpolation."""
    if n_gt == 0:
        return 0.0
    tp_cumsum, fp_cumsum, prec, rec = [], [], [], []
    tp_c = fp_c = 0
    for is_tp, *_ in sorted(tp_fp_list, key=lambda x: -x[1]):
        if is_tp:
            tp_c += 1
        else:
            fp_c += 1
        tp_cumsum.append(tp_c)
        fp_cumsum.append(fp_c)
        prec.append(tp_c / (tp_c + fp_c))
        rec.append(tp_c / n_gt)
    # 11-point
    ap = 0.0
    for t in [i / 10 for i in range(11)]:
        p_at_r = [p for p, r in zip(prec, rec) if r >= t]
        ap += max(p_at_r) / 11 if p_at_r else 0.0
    return ap


def evaluate_yolo(model_path: Path) -> dict:
    """Run YOLO on test images and compute mAP."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))

    per_class_tp_fp = defaultdict(list)
    per_class_n_gt  = defaultdict(int)

    img_files = sorted((TEST_DIR / "images").glob("*.jpg"))
    for img_f in tqdm(img_files, desc="YOLO eval"):
        lbl_f = TEST_DIR / "labels" / (img_f.stem + ".txt")
        img   = Image.open(img_f)
        w, h  = img.size

        gts   = yolo_label_to_boxes(lbl_f, w, h)
        for cls, *_ in gts:
            per_class_n_gt[cls] += 1

        results = model(img_f, conf=CONF_THRESH, verbose=False)[0]
        preds = []
        for box in results.boxes:
            cls = int(box.cls); score = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            preds.append((cls, score, x1, y1, x2, y2))

        matches, _ = match_detections(preds, gts)
        for is_tp, score, cls in matches:
            per_class_tp_fp[cls].append((is_tp, score))

    aps = {}
    for cls in range(len(CLASS_NAMES)):
        ap = compute_ap(per_class_tp_fp.get(cls, []), per_class_n_gt.get(cls, 0))
        aps[CLASS_NAMES[cls]] = round(ap, 4)
    mAP50 = round(sum(aps.values()) / len(aps), 4)
    return {"mAP50": mAP50, "per_class_ap": aps}


def evaluate_grounding_dino() -> dict:
    """Run Grounding DINO on same test images and compute mAP."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "IDEA-Research/grounding-dino-tiny"
    print(f"Loading Grounding DINO ({device}) ...")
    processor = AutoProcessor.from_pretrained(model_id)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    gdino.eval()

    # Use the 3 class names as text prompts
    text_labels = [[c.lower() for c in CLASS_NAMES]]

    per_class_tp_fp = defaultdict(list)
    per_class_n_gt  = defaultdict(int)

    img_files = sorted((TEST_DIR / "images").glob("*.jpg"))
    for img_f in tqdm(img_files, desc="GDINO eval"):
        lbl_f = TEST_DIR / "labels" / (img_f.stem + ".txt")
        img   = Image.open(img_f).convert("RGB")
        w, h  = img.size

        gts = yolo_label_to_boxes(lbl_f, w, h)
        for cls, *_ in gts:
            per_class_n_gt[cls] += 1

        inputs = processor(images=img, text=text_labels, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = gdino(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs, threshold=CONF_THRESH, text_threshold=0.20,
            target_sizes=[(h, w)], text_labels=text_labels,
        )[0]

        preds = []
        for score, raw_lbl, box in zip(results["scores"], results["text_labels"], results["boxes"]):
            raw = str(raw_lbl).lower().strip()
            cls = None
            for i, name in enumerate(CLASS_NAMES):
                if name.lower() in raw or raw in name.lower():
                    cls = i; break
            if cls is None:
                continue
            x1, y1, x2, y2 = box.tolist()
            preds.append((cls, float(score), x1, y1, x2, y2))

        matches, _ = match_detections(preds, gts)
        for is_tp, score, cls in matches:
            per_class_tp_fp[cls].append((is_tp, score))

    aps = {}
    for cls in range(len(CLASS_NAMES)):
        ap = compute_ap(per_class_tp_fp.get(cls, []), per_class_n_gt.get(cls, 0))
        aps[CLASS_NAMES[cls]] = round(ap, 4)
    mAP50 = round(sum(aps.values()) / len(aps), 4)
    return {"mAP50": mAP50, "per_class_ap": aps}


def main():
    yolo_model = ROOT / "models" / "yolo11_dentex.pt"
    if not yolo_model.exists():
        print(f"ERROR: {yolo_model} not found — train first.")
        return

    if not (TEST_DIR / "images").exists():
        print(f"ERROR: Test images not found at {TEST_DIR} — run prepare_dentex.py first.")
        return

    OUT.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Evaluating YOLOv11 ...")
    yolo_res = evaluate_yolo(yolo_model)

    print("\n[2/2] Evaluating Grounding DINO (zero-shot baseline) ...")
    gdino_res = evaluate_grounding_dino()

    comparison = {
        "test_set": "DentexChallenge 2023 test split",
        "iou_threshold": IOU_THRESH,
        "conf_threshold": CONF_THRESH,
        "models": {
            "grounding_dino_zero_shot": gdino_res,
            "yolo11_finetuned": yolo_res,
        },
    }

    out_file = OUT / "model_comparison.json"
    out_file.write_text(json.dumps(comparison, indent=2))

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║          MODEL COMPARISON (DentexChallenge Test)    ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║{'Model':<30} {'mAP@50':>8}  Per-class AP")
    print("╠──────────────────────────────────────────────────────╣")

    for label, res in [("Grounding DINO (zero-shot)", gdino_res), ("YOLOv11 (fine-tuned)", yolo_res)]:
        print(f"║  {label:<28} {res['mAP50']:>7.4f}")
        for cls, ap in res["per_class_ap"].items():
            print(f"║    {cls:<26} {ap:>7.4f}")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\n✓ Comparison saved → {out_file}")


if __name__ == "__main__":
    main()
