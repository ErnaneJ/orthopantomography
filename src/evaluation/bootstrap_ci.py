"""
Per-image bootstrap 95% CI for YOLOv11m mAP@50 on the DentexChallenge test
split, and Wilson 95% CI for the small-sample proportions reported in
Stage 2/3 (Spontaneous Recall bands, high-recall percentage).

Run: python src/evaluation/bootstrap_ci.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT     = Path(__file__).parents[2]
TEST_DIR = ROOT / "data" / "dentex_yolo" / "test"
OUT      = ROOT / "results" / "evaluation"
CLASS_NAMES = ["Caries", "Periapical lesion", "Impacted tooth"]
N_BOOTSTRAP = 2000
SEED = 42


def yolo_label_to_boxes(label_path, img_w, img_h):
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, nw, nh = int(float(parts[0])), *[float(x) for x in parts[1:5]]
        x1 = (cx - nw / 2) * img_w; y1 = (cy - nh / 2) * img_h
        x2 = (cx + nw / 2) * img_w; y2 = (cy + nh / 2) * img_h
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def compute_iou(a, b):
    xa1, ya1, xa2, ya2 = a; xb1, yb1, xb2, yb2 = b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (xa2 - xa1) * (ya2 - ya1) + (xb2 - xb1) * (yb2 - yb1) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, iou_thresh=0.5):
    matched_gt = set(); out = []
    for cls, score, *box in sorted(preds, key=lambda x: -x[1]):
        best_iou, best_i = 0.0, -1
        for i, (gcls, *gbox) in enumerate(gts):
            if i in matched_gt or gcls != cls:
                continue
            iou = compute_iou(box, gbox)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_iou >= iou_thresh and best_i >= 0:
            matched_gt.add(best_i); out.append((True, score, cls))
        else:
            out.append((False, score, cls))
    return out


def compute_ap(tp_fp, n_gt):
    if n_gt == 0:
        return 0.0
    tp_c = fp_c = 0; prec, rec = [], []
    for is_tp, *_ in sorted(tp_fp, key=lambda x: -x[1]):
        if is_tp: tp_c += 1
        else: fp_c += 1
        prec.append(tp_c / (tp_c + fp_c)); rec.append(tp_c / n_gt)
    ap = 0.0
    for t in [i / 10 for i in range(11)]:
        p_at_r = [p for p, r in zip(prec, rec) if r >= t]
        ap += max(p_at_r) / 11 if p_at_r else 0.0
    return ap


def mAP_from_images(image_records):
    """image_records: list of {"preds": [...], "gts": [...]}"""
    per_class_tp_fp = defaultdict(list); per_class_n_gt = defaultdict(int)
    for rec in image_records:
        for cls, *_ in rec["gts"]:
            per_class_n_gt[cls] += 1
        for is_tp, score, cls in match(rec["preds"], rec["gts"]):
            per_class_tp_fp[cls].append((is_tp, score))
    aps = [compute_ap(per_class_tp_fp.get(c, []), per_class_n_gt.get(c, 0)) for c in range(len(CLASS_NAMES))]
    return float(np.mean(aps))


def wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    model_path = ROOT / "models" / "yolo11_dentex.pt"
    from ultralytics import YOLO
    model = YOLO(str(model_path))

    img_files = sorted((TEST_DIR / "images").glob("*.jpg"))
    records = []
    for img_f in tqdm(img_files, desc="Caching YOLO predictions"):
        lbl_f = TEST_DIR / "labels" / (img_f.stem + ".txt")
        img = Image.open(img_f); w, h = img.size
        gts = yolo_label_to_boxes(lbl_f, w, h)
        res = model(img_f, conf=0.001, verbose=False)[0]
        preds = []
        for box in res.boxes:
            cls = int(box.cls); score = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            preds.append((cls, score, x1, y1, x2, y2))
        records.append({"preds": preds, "gts": gts})

    point_estimate = mAP_from_images(records)
    print(f"Point estimate mAP@50 (11-pt AP, custom): {point_estimate:.4f}")

    rng = np.random.default_rng(SEED)
    n = len(records)
    boot_maps = []
    for _ in tqdm(range(N_BOOTSTRAP), desc="Bootstrap resampling"):
        idx = rng.integers(0, n, size=n)
        sample = [records[i] for i in idx]
        boot_maps.append(mAP_from_images(sample))

    lo, hi = np.percentile(boot_maps, [2.5, 97.5])
    print(f"Bootstrap 95% CI (n={N_BOOTSTRAP} resamples, per-image): [{lo:.4f}, {hi:.4f}]")

    # ── Wilson CI for Stage 2/3 proportions ──────────────────────────────────
    s2_dir = ROOT / "results" / "stage2_validations"
    scores = []
    for f in sorted(s2_dir.glob("validation_*.json")):
        d = json.loads(f.read_text())
        sr = d.get("spontaneous_recall")
        if sr is not None:
            scores.append(sr)
    n_sr = len(scores)
    high = sum(1 for s in scores if s >= 0.8)
    wilson_high = wilson_ci(high, n_sr) if n_sr else (0.0, 0.0)

    result = {
        "yolo_map50_bootstrap": {
            "point_estimate": round(point_estimate, 4),
            "ci_95_lower": round(float(lo), 4),
            "ci_95_upper": round(float(hi), 4),
            "n_bootstrap": N_BOOTSTRAP,
            "n_images": n,
            "method": "per-image resampling with replacement; mAP@50 recomputed each resample (11-point AP interpolation, custom implementation, matches compare_baselines.py)",
        },
        "stage2_sr_wilson_ci": {
            "n_evaluable": n_sr,
            "n_high_recall (>=0.8)": high,
            "pct_high_recall": round(high / n_sr * 100, 1) if n_sr else None,
            "wilson_95_ci_pct": [round(wilson_high[0] * 100, 1), round(wilson_high[1] * 100, 1)],
        },
    }
    (OUT / "confidence_intervals.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
