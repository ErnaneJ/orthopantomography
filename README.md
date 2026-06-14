# Automated Dental Pathology Detection in Panoramic Radiographs

> **Paper submitted to [CBEB 2026](https://sbeb.org.br/cbeb2026/) — Brazilian Congress on Biomedical Engineering**
> *Transfer Learning for Dental Pathology Detection in Panoramic Radiographs: YOLOv11 vs. Zero-Shot Foundation Models*
> Ernane Ferreira Rocha Junior, Ignacio Sanchez-Gendriz, Luiz Affonso Guedes — UFRN / CETENE

Research code accompanying the above paper. A three-stage pipeline applied to orthopantomographs (OPGs):

1. **Stage 1 — Detection** YOLOv11m fine-tuned on DentexChallenge 2023 detects Caries, Periapical lesion, and Impacted tooth.
2. **Stage 2 — Spontaneous Recall** Compares detected classes against dentist-written descriptions without prompt injection.
3. **Stage 3 — Report Generation** Gemini 2.5 Flash produces structured pre-clinical reports from the detected findings.

A React + FastAPI web application wraps the full pipeline for interactive use.

---

## Requirements

- Python 3.10+
- Node.js 18+
- `OPENROUTER_API_KEY` for Stage 3 (get one at [openrouter.ai](https://openrouter.ai))
- ~12 GB free disk space if training from scratch (DentexChallenge dataset)
- Apple Silicon (MPS) or CUDA GPU recommended for training; CPU works for inference

---

## Quick start — web application

The web app requires the trained model at `models/yolo11_dentex.pt`. If the file is absent, the backend falls back to Grounding DINO (zero-shot, significantly slower).

```bash
# 1. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. API key
echo "OPENROUTER_API_KEY=sk-or-v1-..." > web/backend/.env

# 3. Frontend dependencies (first run only)
cd web/frontend && npm install && cd ../..

# 4. Start both servers
bash web/start.sh
```

Open **http://localhost:5173**. The backend loads YOLOv11 on startup (allow ~20 seconds).

The script kills any existing processes on ports 8000 and 5173 before starting. Press `Ctrl+C` to stop both servers.

---

## Training from scratch

The full pipeline downloads DentexChallenge 2023, trains YOLOv11n (validation run) and YOLOv11m (full model), evaluates on the test split, compares against Grounding DINO, and runs the private OPG pipeline.

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
bash run_all.sh
```

Estimated time on Apple M5 (16 GB): ~14 hours total (training dominates).

Progress is streamed to stdout and logged to `run_all.log`. To watch from another terminal:

```bash
tail -f run_all.log
```

If the run is interrupted after dataset preparation, use `run_resume.sh` to pick up from step 2 without re-downloading.

### DentexChallenge dataset

The dataset is hosted on Kaggle and requires credentials:

```bash
# Place your kaggle.json at the default location
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Then run the download step
python3 src/data/download_dentex.py
```

---

## Individual pipeline steps

```bash
# Activate environment first
source .venv/bin/activate

# Dataset preparation (COCO → YOLO format, 70/15/15 split, augmentation)
python3 src/data/prepare_dentex.py

# Train YOLOv11m (saves best weights to results/training/opg_yolo_medium/)
python3 src/training/train_yolo.py --model yolo11m.pt --epochs 100 --batch 8 --name opg_yolo_medium
cp results/training/opg_yolo_medium/weights/best.pt models/yolo11_dentex.pt

# Evaluate on DentexChallenge test split (mAP@50, mAP@50:95, per-class AP)
python3 src/evaluation/evaluate_dentex.py

# Baseline comparison: YOLOv11 vs Grounding DINO (zero-shot)
python3 src/evaluation/compare_baselines.py

# Full pipeline on private OPGs (all stages)
python3 src/run_pipeline.py

# Skip LLM stage (no API key needed)
python3 src/run_pipeline.py --skip-stage3

# Run specific stages only
python3 src/run_pipeline.py --stages 1 2

# Generate paper figures
python3 src/generate_paper_figures.py
# Output: results/figures/fig1_pipeline_architecture.png ... fig6_nlp_metrics.png
```

---

## Web application — API reference

The FastAPI backend exposes a REST API at `http://localhost:8000`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Model status |
| `POST` | `/api/analyses` | Upload OPG, trigger detection + report |
| `GET` | `/api/analyses` | List all analyses |
| `GET` | `/api/analyses/{id}` | Get analysis detail |
| `GET` | `/api/analyses/{id}/pdf` | Export report as PDF |
| `POST` | `/api/annotations` | Add manual annotation box |
| `DELETE` | `/api/annotations/{id}` | Remove annotation |
| `POST` | `/api/analyses/{id}/enrich` | Re-run LLM report with updated detections |

Interactive docs at `http://localhost:8000/docs` (Swagger UI).

---

## Project structure

```
├── src/
│   ├── data/
│   │   ├── download_dentex.py      # Download DentexChallenge 2023 from Kaggle
│   │   └── prepare_dentex.py       # COCO→YOLO, split, augmentation
│   ├── training/
│   │   └── train_yolo.py           # YOLOv11 fine-tuning
│   ├── evaluation/
│   │   ├── evaluate_dentex.py      # mAP evaluation on test split
│   │   └── compare_baselines.py    # YOLOv11 vs Grounding DINO comparison
│   ├── pipeline/
│   │   ├── stage1_yolo.py          # Detection (YOLOv11)
│   │   ├── stage1_detection.py     # Detection fallback (Grounding DINO)
│   │   ├── stage2_validation.py    # Spontaneous recall evaluation
│   │   ├── stage3_report.py        # LLM report generation (Gemini 2.5 Flash)
│   │   ├── stage4_metrics.py       # NLP metric computation
│   │   ├── config.py
│   │   └── utils.py
│   ├── run_pipeline.py             # Run pipeline on private OPGs
│   └── generate_paper_figures.py   # Generate all paper figures
│
├── web/
│   ├── backend/                    # FastAPI + SQLAlchemy + SQLite
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── database.py
│   │   ├── routers/                # analyses, detections, annotations, enrich, pdf
│   │   └── services/               # detection, image, report, validation, pdf
│   ├── frontend/                   # React 19 + TypeScript + Tailwind + Vite
│   │   ├── src/
│   │   │   ├── pages/              # DetailPage, ListPage, StatsPage
│   │   │   ├── components/         # AnnotationCanvas, ReportTabs, FindingsList, ...
│   │   │   ├── api/
│   │   │   ├── types/
│   │   │   └── utils/
│   │   └── package.json
│   └── start.sh                    # Start backend + frontend
│
├── models/                         # Trained weights (not tracked in git)
│   └── yolo11_dentex.pt
│
├── data/
│   └── Experiments/
│       └── EDA.ipynb               # Exploratory data analysis
│
├── results/
│   ├── evaluation/                 # dentex_test_metrics.json, model_comparison.json
│   ├── metrics/                    # all_metrics.json (aggregated pipeline metrics)
│   └── figures/                    # Paper figures (fig1_ through fig6_)
│
├── paper_cbeb2026.tex              # IEEE-format paper (CBEB 2026)
├── references.bib                  # BibTeX bibliography
├── requirements.txt                # Python dependencies
├── run_all.sh                      # Full pipeline (download → train → evaluate)
└── run_resume.sh                   # Resume from dataset preparation (skip download)
```

---

## Key results

Evaluated on the DentexChallenge 2023 test split (n = 103 images, IoU = 0.5):

| Model | Caries AP@50 | Periapical AP@50 | Impacted AP@50 | mAP@50 |
|---|:---:|:---:|:---:|:---:|
| Grounding DINO (zero-shot) | 0.000 | 0.000 | 0.000 | 0.000 |
| YOLOv11m (fine-tuned) | 0.303 | 0.095 | 0.566 | 0.499 |

YOLOv11m: mAP@50:95 = 0.321, Precision = 0.583, Recall = 0.550.

Private OPG pipeline (n = 50 images):
- Stage 2 spontaneous recall: mean 88.3% (SD 30.8%, n = 30 evaluable images)
- Stage 3 BERTScore F1: 0.780 (RoBERTa-large, n = 50 reports)

---

## Citation

This repository accompanies a paper submitted to CBEB 2026. If you use this code, please cite:

```bibtex
@inproceedings{rocha2026opg,
  author    = {Rocha Junior, Ernane Ferreira and S{\'a}nchez-Gendriz, Ignacio and Guedes, Luiz Affonso},
  title     = {Transfer Learning for Dental Pathology Detection in Panoramic Radiographs:
               {YOLOv11} vs. Zero-Shot Foundation Models},
  booktitle = {Proceedings of the Brazilian Congress on Biomedical Engineering (CBEB)},
  year      = {2026}
}
```

---

## Notes

- The private OPG dataset (50 clinical images, descriptions, and audio) is not included in this repository.
- The DentexChallenge 2023 dataset is CC0 licensed. See [dentex.grand-challenge.org](https://dentex.grand-challenge.org/).
- Training defaults to the Apple MPS backend. On Linux with CUDA, Ultralytics selects the GPU automatically.
- The `models/` directory is git-ignored. You must either train the model or obtain `yolo11_dentex.pt` separately before running the web app in YOLO mode.
