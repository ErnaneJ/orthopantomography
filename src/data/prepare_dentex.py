"""
Prepare DentexChallenge 2023 for YOLOv11 training.

JSON structure (discovered):
  categories_3 = disease classes:
    id 0 → "Impacted"         → YOLO 2 "Impacted tooth"
    id 1 → "Caries"           → YOLO 0 "Caries"
    id 2 → "Periapical Lesion"→ YOLO 1 "Periapical lesion"
    id 3 → "Deep Caries"      → YOLO 0 "Caries" (merged)
  annotations use category_id_3 for disease labels.

Run: python src/data/prepare_dentex.py
"""
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT     = Path(__file__).parents[2]
RAW      = ROOT / "data" / "dentex_raw" / "training_data" / "training_data" / "quadrant-enumeration-disease"
OUT      = ROOT / "data" / "dentex_yolo"
YAML_OUT = ROOT / "src" / "training" / "dentex.yaml"

CLASS_NAMES = ["Caries", "Periapical lesion", "Impacted tooth"]

# category_id_3 → our YOLO class ID
CAT3_TO_YOLO = {
    0: 2,   # Impacted       → Impacted tooth
    1: 0,   # Caries         → Caries
    2: 1,   # Periapical Lesion → Periapical lesion
    3: 0,   # Deep Caries    → Caries (merged)
}

RARE_CLASSES = {1, 2}   # Periapical lesion (only 158 samples), Impacted tooth (604)
SEED = 42
random.seed(SEED); np.random.seed(SEED)


# ── Augmentation ─────────────────────────────────────────────────────────────
try:
    import albumentations as A
    _AUG = A.Compose([
        A.CLAHE(clip_limit=3.0, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-5, 5),
                 border_mode=cv2.BORDER_CONSTANT, p=0.4),
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"],
                                min_visibility=0.25))
    HAS_ALB = True
except ImportError:
    HAS_ALB = False
    print("albumentations not found — basic augmentation only")


def _augment(img, bboxes, class_ids):
    if HAS_ALB:
        try:
            r = _AUG(image=img, bboxes=bboxes, class_labels=class_ids)
            return r["image"], r["bboxes"], r["class_labels"]
        except Exception:
            pass
    # Fallback: CLAHE + random flip
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
        bboxes = [(1 - cx, cy, w, h) for cx, cy, w, h in bboxes]
    return img, bboxes, class_ids


def coco_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = max(0.001, min(0.999, (x + w / 2) / img_w))
    cy = max(0.001, min(0.999, (y + h / 2) / img_h))
    nw = max(0.001, min(0.999, w / img_w))
    nh = max(0.001, min(0.999, h / img_h))
    return cx, cy, nw, nh


def main():
    json_path = RAW / "train_quadrant_enumeration_disease.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found — run download_dentex.py first.")
        return

    print(f"Loading {json_path.name} ...")
    with open(json_path) as f:
        coco = json.load(f)

    # Confirm categories_3 structure
    print("Disease categories (categories_3):")
    for cat in coco["categories_3"]:
        yolo_id = CAT3_TO_YOLO.get(cat["id"])
        print(f"  id={cat['id']} '{cat['name']}' → YOLO {yolo_id} ({CLASS_NAMES[yolo_id] if yolo_id is not None else 'SKIP'})")

    # Build image index
    img_info = {img["id"]: img for img in coco["images"]}

    # Group annotations per image using category_id_3
    img_anns: dict[int, list] = defaultdict(list)
    skipped = 0
    for ann in coco["annotations"]:
        cat3 = ann.get("category_id_3")
        if cat3 is None:
            skipped += 1; continue
        yolo_cls = CAT3_TO_YOLO.get(cat3)
        if yolo_cls is None:
            skipped += 1; continue
        img_anns[ann["image_id"]].append((yolo_cls, ann["bbox"]))

    print(f"\nImages with disease annotations: {len(img_anns)} / {len(img_info)}")
    print(f"Skipped annotations: {skipped}")

    class_counts = defaultdict(int)
    for anns in img_anns.values():
        for cls, _ in anns:
            class_counts[cls] += 1
    print("\nClass distribution (raw):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {class_counts[i]} annotations")

    # Stratified split 70/15/15
    img_ids = list(img_anns.keys())
    random.shuffle(img_ids)
    n = len(img_ids)
    n_train = int(n * 0.70); n_val = int(n * 0.15)
    splits = {
        "train": img_ids[:n_train],
        "val":   img_ids[n_train:n_train + n_val],
        "test":  img_ids[n_train + n_val:],
    }
    print(f"\nSplit: train={len(splits['train'])} | val={len(splits['val'])} | test={len(splits['test'])}")

    for sp in splits:
        (OUT / sp / "images").mkdir(parents=True, exist_ok=True)
        (OUT / sp / "labels").mkdir(parents=True, exist_ok=True)

    xray_dir = RAW / "xrays"
    stats = defaultdict(int)
    aug_count = 0

    for split, ids in splits.items():
        print(f"\nProcessing {split} ...")
        for img_id in tqdm(ids, desc=split):
            info  = img_info[img_id]
            fname = Path(info["file_name"]).name
            src   = xray_dir / fname
            if not src.exists():
                src = xray_dir / info["file_name"]
            if not src.exists():
                continue

            w_img, h_img = info["width"], info["height"]
            anns = img_anns[img_id]
            classes_in_img = set()
            yolo_lines, bboxes_aug, cls_aug = [], [], []

            for cls_id, coco_bbox in anns:
                cx, cy, nw, nh = coco_to_yolo(coco_bbox, w_img, h_img)
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                bboxes_aug.append((cx, cy, nw, nh))
                cls_aug.append(cls_id)
                classes_in_img.add(cls_id)
                stats[cls_id] += 1

            stem  = src.stem
            dst_j = OUT / split / "images" / f"{stem}.jpg"
            dst_l = OUT / split / "labels" / f"{stem}.txt"

            img_cv = cv2.imread(str(src))
            if img_cv is not None:
                cv2.imwrite(str(dst_j), img_cv, [cv2.IMWRITE_JPEG_QUALITY, 90])
            else:
                shutil.copy(src, dst_j)
            dst_l.write_text("\n".join(yolo_lines))

            # Oversample rare classes on train (2 augmented copies → 3× total)
            if split == "train" and classes_in_img & RARE_CLASSES and img_cv is not None:
                for aug_i in range(2):
                    aug_img, aug_bboxes, aug_cls = _augment(img_cv.copy(), bboxes_aug[:], cls_aug[:])
                    if not aug_bboxes:
                        continue
                    aug_stem = f"{stem}_aug{aug_i}"
                    cv2.imwrite(
                        str(OUT / split / "images" / f"{aug_stem}.jpg"),
                        aug_img, [cv2.IMWRITE_JPEG_QUALITY, 88]
                    )
                    aug_lines = [
                        f"{c} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
                        for c, (cx, cy, nw, nh) in zip(aug_cls, aug_bboxes)
                    ]
                    (OUT / split / "labels" / f"{aug_stem}.txt").write_text("\n".join(aug_lines))
                    aug_count += 1

    print(f"\nAugmented images added: {aug_count}")
    print("\nFinal annotation counts (originals only):")
    total = 0
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {stats[i]}"); total += stats[i]
    print(f"  TOTAL: {total}")

    # Write dentex.yaml
    YAML_OUT.parent.mkdir(parents=True, exist_ok=True)
    YAML_OUT.write_text(
        f"# DentexChallenge 2023 — 3 disease classes\n"
        f"path: {OUT.resolve()}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"test:  test/images\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    print(f"\n✓ dentex.yaml → {YAML_OUT}")
    print(f"✓ Dataset ready → {OUT}")
    print("\nNext: python src/training/train_yolo.py")


if __name__ == "__main__":
    main()
