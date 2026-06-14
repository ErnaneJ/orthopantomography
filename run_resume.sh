#!/bin/bash
set -e
cd "$(dirname "$0")"
LOG="run_all.log"
exec >> "$LOG" 2>&1

echo ""
echo "============================================================"
echo "  RESUME from step 2 — $(date)"
echo "============================================================"

step() { echo ""; echo "────────────────────────────────────────"; echo "[$1/7] $2"; echo "Started: $(date)"; }
ok()   { echo "✓ Done: $(date)"; }
fail() { echo "✗ FAILED at step $1 — check run_all.log"; exit 1; }

step 2 "Prepare dataset"
python3 src/data/prepare_dentex.py || fail 2
ok

step 3 "Training YOLOv11n — 50 epochs (~2h, pipeline validation)"
python3 src/training/train_yolo.py --model yolo11n.pt --epochs 50 --batch 8 --name opg_yolo_nano || fail 3
ok

step 4 "Training YOLOv11m — 100 epochs (~12h, full model)"
python3 src/training/train_yolo.py --model yolo11m.pt --epochs 100 --batch 8 --name opg_yolo_medium || fail 4
cp results/training/opg_yolo_medium/weights/best.pt models/yolo11_dentex.pt
echo "✓ Best model → models/yolo11_dentex.pt"
ok

step 5 "Evaluate mAP on DentexChallenge test split"
python3 src/evaluation/evaluate_dentex.py || fail 5
ok

step 6 "Baseline comparison: YOLOv11 vs Grounding DINO"
python3 src/evaluation/compare_baselines.py || fail 6
ok

step 7 "Pipeline on 50 private OPGs"
python3 src/run_pipeline.py --skip-stage3 || fail 7
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo "Running Stage 3 (LLM)..."
    python3 src/run_pipeline.py --stages 3 4 || true
else
    python3 src/run_pipeline.py --stages 4 || true
fi
ok

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results: results/  |  Model: models/yolo11_dentex.pt"
echo "============================================================"
