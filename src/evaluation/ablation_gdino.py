"""
Grounding DINO ablation: prompt formulation, model variant (Tiny vs Base),
and IoU threshold sweep, on the DentexChallenge 2023 test split.

Addresses reviewer requests: (1) a single fixed prompt is not enough evidence
that zero-shot grounding fails outright — test multiple phrasings; (2) report
AP at IoU thresholds below the clinical 0.5 standard to characterize *how*
close the candidate boxes get, rather than asserting it without numbers.

Raw detections are cached per (model, prompt) config so the IoU sweep does not
require re-running inference.

Run: python src/evaluation/ablation_gdino.py
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
CACHE    = OUT / "gdino_ablation_raw.json"

CLASS_NAMES = ["Caries", "Periapical lesion", "Impacted tooth"]
CONF_THRESH = 0.25
TEXT_THRESH = 0.20
IOU_SWEEP = [0.10, 0.25, 0.50]

PROMPTS = {
    "class_name":  ["caries", "periapical lesion", "impacted tooth"],
    "descriptive": ["dental caries", "periapical radiolucency", "impacted tooth"],
    "phrase":      ["a tooth with caries", "a periapical lesion on a tooth root", "an impacted tooth"],
}

MODELS = {
    "tiny": "IDEA-Research/grounding-dino-tiny",
    "base": "IDEA-Research/grounding-dino-base",
}


def compute_iou(box_a, box_b):
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


def match_detections(preds: list, gts: list, iou_thresh: float):
    matched_gt = set()
    results = []
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
    return results


def compute_ap(tp_fp_list: list, n_gt: int) -> float:
    if n_gt == 0:
        return 0.0
    tp_c = fp_c = 0
    prec, rec = [], []
    for is_tp, *_ in sorted(tp_fp_list, key=lambda x: -x[1]):
        if is_tp:
            tp_c += 1
        else:
            fp_c += 1
        prec.append(tp_c / (tp_c + fp_c))
        rec.append(tp_c / n_gt)
    ap = 0.0
    for t in [i / 10 for i in range(11)]:
        p_at_r = [p for p, r in zip(prec, rec) if r >= t]
        ap += max(p_at_r) / 11 if p_at_r else 0.0
    return ap


def run_inference(model_id: str, prompt: list) -> dict:
    """Returns {img_stem: {"gts": [...], "preds": [(cls, score, x1,y1,x2,y2), ...]}}"""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    gdino.eval()

    text_labels = [prompt]
    raw = {}
    img_files = sorted((TEST_DIR / "images").glob("*.jpg"))
    for img_f in tqdm(img_files, desc=f"{model_id.split('/')[-1]} | {prompt[0]}"):
        lbl_f = TEST_DIR / "labels" / (img_f.stem + ".txt")
        img = Image.open(img_f).convert("RGB")
        w, h = img.size
        gts = yolo_label_to_boxes(lbl_f, w, h)

        inputs = processor(images=img, text=text_labels, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = gdino(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs, threshold=CONF_THRESH, text_threshold=TEXT_THRESH,
            target_sizes=[(h, w)], text_labels=text_labels,
        )[0]

        preds = []
        for score, raw_lbl, box in zip(results["scores"], results["text_labels"], results["boxes"]):
            raw_l = str(raw_lbl).lower().strip()
            cls = None
            for i, name in enumerate(prompt):
                if name.lower() in raw_l or raw_l in name.lower():
                    cls = i; break
            if cls is None:
                continue
            x1, y1, x2, y2 = box.tolist()
            preds.append((cls, float(score), x1, y1, x2, y2))

        raw[img_f.stem] = {"gts": gts, "preds": preds}

    del gdino, processor
    return raw


def score_config(raw: dict, iou_thresh: float) -> dict:
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
    if not (TEST_DIR / "images").exists():
        print(f"ERROR: Test images not found at {TEST_DIR} — run prepare_dentex.py first.")
        return

    OUT.mkdir(parents=True, exist_ok=True)

    # 1) Prompt ablation on Tiny
    all_raw = {}
    for prompt_name, prompt in PROMPTS.items():
        key = f"tiny__{prompt_name}"
        print(f"\n=== {key} ===")
        all_raw[key] = run_inference(MODELS["tiny"], prompt)

    # 2) Base variant with the standard class-name prompt
    key = "base__class_name"
    print(f"\n=== {key} ===")
    all_raw[key] = run_inference(MODELS["base"], PROMPTS["class_name"])

    # 3) Score every config at every IoU threshold in the sweep
    report = {"iou_sweep": IOU_SWEEP, "prompts": PROMPTS, "configs": {}}
    for key, raw in all_raw.items():
        report["configs"][key] = {
            f"iou_{iou}": score_config(raw, iou) for iou in IOU_SWEEP
        }

    OUT_FILE = OUT / "gdino_ablation.json"
    OUT_FILE.write_text(json.dumps(report, indent=2))

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║           GROUNDING DINO ABLATION — DentexChallenge Test      ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    for key, cfg in report["configs"].items():
        print(f"\n  {key}")
        for iou in IOU_SWEEP:
            r = cfg[f"iou_{iou}"]
            print(f"    IoU={iou:.2f}  mAP={r['mAP']:.4f}  per-class={r['per_class_ap']}")
    print(f"\n✓ Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
