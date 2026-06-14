"""
YOLOv11 training on DentexChallenge 2023 — MacBook M5 (MPS).

Usage:
  # Quick prototype (yolo11n, 50 epochs, ~2h on M5)
  python src/training/train_yolo.py

  # Full training (yolo11m, 100 epochs, ~12h overnight on M5)
  python src/training/train_yolo.py --model yolo11m --epochs 100

  # Resume interrupted training
  python src/training/train_yolo.py --resume results/training/opg_yolo/weights/last.pt
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).parents[2]
YAML = ROOT / "src" / "training" / "dentex.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="yolo11n.pt",
                        help="Base checkpoint: yolo11n.pt | yolo11s.pt | yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=8,
                        help="Batch size — M5 15GB handles 8 for 640px")
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--resume", default=None,
                        help="Path to last.pt to resume training")
    parser.add_argument("--name",   default="opg_yolo",
                        help="Run name under results/training/")
    args = parser.parse_args()

    from ultralytics import YOLO
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  {args.model}  |  Epochs: {args.epochs}  |  Batch: {args.batch}")

    if args.resume:
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    model = YOLO(args.model)
    model.train(
        data=str(YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=4,
        project=str(ROOT / "results" / "training"),
        name=args.name,
        exist_ok=True,
        # X-ray specific: disable colour augmentations (grayscale)
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        # Geometric augmentations
        flipud=0.0,    # no vertical flip (teeth have orientation)
        fliplr=0.5,    # horizontal flip OK (bilateral anatomy)
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.1,
        # Early stopping
        patience=20,
        # Save checkpoints every 10 epochs
        save_period=10,
        # Optimiser
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        warmup_epochs=3,
        # Conf/NMS
        conf=0.25,
        iou=0.7,
        verbose=True,
    )

    # Copy best weights to models/
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    best = ROOT / "results" / "training" / args.name / "weights" / "best.pt"
    if best.exists():
        import shutil
        dest = models_dir / "yolo11_dentex.pt"
        shutil.copy(best, dest)
        print(f"\n✓ Best weights → {dest}")
        print("Próximo passo: python src/evaluation/evaluate_dentex.py")
    else:
        print(f"\nBest weights not found at {best} — check training logs.")


if __name__ == "__main__":
    main()
