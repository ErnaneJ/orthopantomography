#!/bin/bash
# GAIA SLURM — opcional, só se precisar de treino mais rápido
#SBATCH --job-name=opg_yolo
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/yolo_%j.out

set -e
module load python/3.11 cuda/12.1 2>/dev/null || true

cd $HOME/orthopantomography
source venv/bin/activate 2>/dev/null || source .venv/bin/activate

python src/training/train_yolo.py \
  --model yolo11m.pt \
  --epochs 100 \
  --batch 32 \
  --name opg_yolo_gaia

echo "Done. Best weights → models/yolo11_dentex.pt"
