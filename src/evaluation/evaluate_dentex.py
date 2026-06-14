"""
Evaluate trained YOLOv11 on DentexChallenge 2023 test split.
Outputs mAP@50, mAP@50:95, Precision, Recall per class.

Run: python src/evaluation/evaluate_dentex.py [--model models/yolo11_dentex.pt]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parents[2]
YAML = ROOT / "src" / "training" / "dentex.yaml"
OUT  = ROOT / "results" / "evaluation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "yolo11_dentex.pt"))
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run training first: python src/training/train_yolo.py")
        return

    from ultralytics import YOLO

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    print("Evaluating on test split ...")
    metrics = model.val(
        data=str(YAML),
        split="test",
        conf=0.25,
        iou=0.5,
        verbose=True,
    )

    # Extract per-class results
    class_names = model.names  # {0: 'Caries', 1: 'Periapical lesion', 2: 'Impacted tooth'}

    results = {
        "model": str(model_path.name),
        "split": "test",
        "aggregate": {
            "mAP50":    round(float(metrics.box.map50),  4),
            "mAP50_95": round(float(metrics.box.map),    4),
            "precision": round(float(metrics.box.mp),    4),
            "recall":    round(float(metrics.box.mr),    4),
        },
        "per_class": {},
    }

    # Per-class metrics
    try:
        for i, name in class_names.items():
            results["per_class"][name] = {
                "mAP50":     round(float(metrics.box.maps[i]), 4),
                "precision": round(float(metrics.box.p[i]),    4) if hasattr(metrics.box, 'p') else None,
                "recall":    round(float(metrics.box.r[i]),    4) if hasattr(metrics.box, 'r') else None,
            }
    except Exception as e:
        print(f"Note: Per-class detail not available ({e})")

    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / "dentex_test_metrics.json"
    out_file.write_text(json.dumps(results, indent=2))

    print("\n╔════════════════════════════════════════════╗")
    print("║   DENTEX TEST SET — YOLO EVALUATION        ║")
    print("╠════════════════════════════════════════════╣")
    print(f"║  mAP@50       : {results['aggregate']['mAP50']:.4f}")
    print(f"║  mAP@50:95    : {results['aggregate']['mAP50_95']:.4f}")
    print(f"║  Precision    : {results['aggregate']['precision']:.4f}")
    print(f"║  Recall       : {results['aggregate']['recall']:.4f}")
    if results["per_class"]:
        print("╠────────────────────────────────────────────╣")
        for cls, m in results["per_class"].items():
            print(f"║  {cls:<22} mAP50: {m['mAP50']:.4f}")
    print("╚════════════════════════════════════════════╝")
    print(f"\n✓ Results saved → {out_file}")
    print("Próximo passo: python src/evaluation/compare_baselines.py")


if __name__ == "__main__":
    main()
