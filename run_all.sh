#!/bin/bash
# Master pipeline — executa tudo em sequência.
# Log completo em: run_all.log
# Para ver progresso: tail -f run_all.log

set -e
cd "$(dirname "$0")"

LOG="run_all.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "  OPG FULL PIPELINE — $(date)"
echo "============================================================"

step() { echo ""; echo "────────────────────────────────────────"; echo "[$1/7] $2"; echo "Started: $(date)"; }
ok()   { echo "✓ Done: $(date)"; }
fail() { echo "✗ FAILED at step $1 — check $LOG"; exit 1; }

# ── 1. Download DentexChallenge ──────────────────────────────────────────────
step 1 "Download DentexChallenge 2023 (~11 GB)"
python3 src/data/download_dentex.py || fail 1
ok

# ── 2. Prepare dataset ───────────────────────────────────────────────────────
step 2 "Prepare dataset (COCO→YOLO, split, augmentation)"
python3 src/data/prepare_dentex.py || fail 2
ok

# ── 3. Quick training — validate pipeline ────────────────────────────────────
step 3 "Training YOLOv11n — 50 epochs (pipeline validation, ~2h)"
python3 src/training/train_yolo.py \
    --model yolo11n.pt \
    --epochs 50 \
    --batch 8 \
    --name opg_yolo_nano || fail 3
ok

# ── 4. Full training ─────────────────────────────────────────────────────────
step 4 "Training YOLOv11m — 100 epochs (full model, ~12h)"
python3 src/training/train_yolo.py \
    --model yolo11m.pt \
    --epochs 100 \
    --batch 8 \
    --name opg_yolo_medium || fail 4

# Copy best model
cp results/training/opg_yolo_medium/weights/best.pt models/yolo11_dentex.pt
echo "✓ Best model → models/yolo11_dentex.pt"
ok

# ── 5. Evaluate on DentexChallenge test split ────────────────────────────────
step 5 "Evaluate YOLOv11 on DentexChallenge test split (mAP)"
python3 src/evaluation/evaluate_dentex.py || fail 5
ok

# ── 6. Baseline comparison ───────────────────────────────────────────────────
step 6 "Baseline comparison: YOLOv11 vs Grounding DINO"
python3 src/evaluation/compare_baselines.py || fail 6
ok

# ── 7. Run full pipeline on 50 private OPGs ──────────────────────────────────
step 7 "Run pipeline on 50 OPGs (Stage 1→2→3→4)"
python3 src/run_pipeline.py --skip-stage3 || fail 7
# Stage 3 (LLM) separately — needs API key
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo "Running Stage 3 (LLM report generation)..."
    python3 src/run_pipeline.py --stages 3 4 || true
else
    echo "⚠  OPENROUTER_API_KEY not set — skipping Stage 3 (LLM reports)"
    python3 src/run_pipeline.py --stages 4 || true
fi
ok

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
echo "  Results in: results/"
echo "  Model at:   models/yolo11_dentex.pt"
echo "  Log:        run_all.log"
echo "============================================================"
