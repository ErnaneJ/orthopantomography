"""
Generate all figures for the CBEB 2026 paper.
Run: python src/generate_paper_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).parents[1]
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Common style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})

GRAY   = "#333333"
BLUE   = "#1565C0"
RED    = "#C62828"
GREEN  = "#2E7D32"
ORANGE = "#E65100"
PURPLE = "#6A1B9A"
LIGHT  = "#E3F2FD"


# ── Fig 1 — Pipeline architecture ─────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis("off")

    # 5 blocks: Dataset Prep | YOLOv11 Training | Stage 1 | Stage 2 | Stage 3
    blocks = [
        ("Data\nPreparation",
         "DentexChallenge 2023\n1,154 images\nCaries / Periapical /\nImpacted tooth\n70/15/15 split\nOversampling (rare)",
         "#E3F2FD", "#1565C0"),
        ("YOLOv11\nTraining",
         "YOLOv11m pretrained\nAdamW  lr=10⁻³\nbatch=8  img=640 px\n100 epochs  MPS (M5)\nAugment: CLAHE, flip,\nbrightness, affine",
         "#E8F5E9", "#2E7D32"),
        ("Stage 1\nDetection",
         "YOLOv11 fine-tuned\non 50 OPGs\nmAP@50 = 0.499\n(DentexChallenge test)\n466 det. / 50 images\nConf ≥ 0.25",
         "#FFF8E1", "#E65100"),
        ("Stage 2\nSpontaneous Recall",
         "Dentist descriptions\n→ class mention check\nSR = detected ∩ mentioned\n/ mentioned\nMean SR = 88.3%\n30 evaluable images",
         "#F3E5F5", "#6A1B9A"),
        ("Stage 3\nLLM Report",
         "Gemini 2.5 Flash\nvia OpenRouter API\nImage + detections\n5-section structured report\nBERTScore F1 = 0.780\n50 reports",
         "#FCE4EC", "#C62828"),
    ]

    n = len(blocks)
    block_w, block_h = 2.0, 3.2
    gap = 0.45
    total_w = n * block_w + (n - 1) * gap
    x0 = (13 - total_w) / 2

    for i, (title, body, fill, edge) in enumerate(blocks):
        x = x0 + i * (block_w + gap)
        y = 0.6
        rect = mpatches.FancyBboxPatch(
            (x, y), block_w, block_h,
            boxstyle="round,pad=0.08",
            facecolor=fill, edgecolor=edge, linewidth=1.8,
            transform=ax.transData, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x + block_w / 2, y + block_h - 0.32, title,
                ha="center", va="top", fontsize=10, fontweight="bold",
                color=edge, transform=ax.transData, zorder=3)
        ax.text(x + block_w / 2, y + block_h * 0.52, body,
                ha="center", va="center", fontsize=7.5, color=GRAY,
                transform=ax.transData, zorder=3, linespacing=1.35)

        # Arrow to next block
        if i < n - 1:
            ax.annotate(
                "", xy=(x + block_w + gap, y + block_h / 2),
                xytext=(x + block_w, y + block_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=GRAY,
                                lw=1.4, mutation_scale=14),
                transform=ax.transData, zorder=4,
            )

    # Evaluation arrow going down from Stage 1
    eval_x = x0 + 2 * (block_w + gap) + block_w / 2
    ax.annotate(
        "", xy=(eval_x, 0.35), xytext=(eval_x, 0.6),
        arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.2, mutation_scale=12),
        transform=ax.transData, zorder=4,
    )
    ax.text(eval_x, 0.22, "Evaluation on\nDentexChallenge\ntest set",
            ha="center", va="center", fontsize=7.5, color=GRAY,
            style="italic", transform=ax.transData)

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.5)
    fig.suptitle(
        "Fig. 1. Three-stage pipeline for panoramic dental radiograph analysis.",
        fontsize=10, y=0.02, color=GRAY,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = FIGS / "fig1_pipeline_architecture.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  fig1_pipeline_architecture.png")


# ── Fig 2 — Training curves (mAP50 over epochs) ───────────────────────────────
def fig_training_curves():
    csv_nano   = ROOT / "results/training/opg_yolo_nano/results.csv"
    csv_medium = ROOT / "results/training/opg_yolo_medium/results.csv"

    def read_csv(p):
        epochs, map50, map50_95 = [], [], []
        with open(p) as f:
            headers = [h.strip() for h in f.readline().split(",")]
            ep_col  = headers.index("epoch")
            m50_col = headers.index("metrics/mAP50(B)")
            m95_col = headers.index("metrics/mAP50-95(B)")
            for line in f:
                parts = line.strip().split(",")
                if not parts[0].strip():
                    continue
                epochs.append(int(float(parts[ep_col])))
                map50.append(float(parts[m50_col]))
                map50_95.append(float(parts[m95_col]))
        return np.array(epochs), np.array(map50), np.array(map50_95)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, csv_path, label, color in [
        (axes[0], csv_nano,   "YOLOv11n (nano)", BLUE),
        (axes[1], csv_medium, "YOLOv11m (medium)", GREEN),
    ]:
        if not csv_path.exists():
            ax.text(0.5, 0.5, "Data not found", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            continue
        epochs, map50, map50_95 = read_csv(csv_path)
        ax.plot(epochs, map50,    color=color,  lw=2,   label="mAP@50",    zorder=3)
        ax.plot(epochs, map50_95, color=color,  lw=1.5, linestyle="--",
                alpha=0.6, label="mAP@50-95", zorder=3)
        best_idx = np.argmax(map50)
        ax.scatter(epochs[best_idx], map50[best_idx], color=RED, s=60, zorder=4)
        ax.annotate(
            f"Best: {map50[best_idx]:.3f}\n@ epoch {epochs[best_idx]}",
            xy=(epochs[best_idx], map50[best_idx]),
            xytext=(epochs[best_idx] + max(1, len(epochs) * 0.08),
                    map50[best_idx] - 0.07),
            fontsize=8, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.9),
        )
        ax.fill_between(epochs, map50, alpha=0.07, color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP")
        ax.set_title(label)
        ax.set_ylim(0, 0.75)
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    fig.suptitle("Fig. 2. Validation mAP curves during fine-tuning on DentexChallenge 2023.",
                 fontsize=10, y=0.02, color=GRAY)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIGS / "fig2_training_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_training_curves.png")


# ── Fig 3 — Model comparison (Grounding DINO vs YOLOv11) ──────────────────────
def fig_model_comparison():
    comp = {
        "Grounding DINO\n(zero-shot)": {
            "Caries": 0.0, "Periapical\nlesion": 0.0, "Impacted\ntooth": 0.0,
            "Mean mAP@50": 0.0,
        },
        "YOLOv11m\n(fine-tuned)": {
            "Caries": 0.3034, "Periapical\nlesion": 0.0950, "Impacted\ntooth": 0.5657,
            "Mean mAP@50": 0.4992,
        },
    }

    classes = ["Caries", "Periapical\nlesion", "Impacted\ntooth", "Mean mAP@50"]
    x = np.arange(len(classes))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (model, vals) in enumerate(comp.items()):
        offset = (i - 0.5) * width
        color  = [RED, BLUE][i]
        bars = ax.bar(x + offset, [vals[c] for c in classes],
                      width, label=model, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, cls in zip(bars, classes):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.012, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8,
                    color=bar.get_facecolor(), fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Average Precision @ IoU 0.5")
    ax.set_title("Zero-shot vs fine-tuned detection on DentexChallenge 2023 test set")
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False)
    ax.axvline(2.5, color="gray", lw=0.8, linestyle=":", alpha=0.5)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    fig.suptitle("Fig. 3. Per-class and mean mAP@50 comparison (test split, IoU=0.5, conf=0.25).",
                 fontsize=10, y=0.02, color=GRAY)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIGS / "fig3_model_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_model_comparison.png")


# ── Fig 4 — Detection frequency on private OPGs ───────────────────────────────
def fig_detection_frequency():
    summary_f = ROOT / "results/stage1_detections/summary.json"
    if not summary_f.exists():
        print("  fig4 skipped — no stage1 summary")
        return
    summary = json.loads(summary_f.read_text())
    freq = summary.get("class_frequency", {})
    if not freq:
        print("  fig4 skipped — empty class_frequency")
        return

    labels = list(freq.keys())
    values = list(freq.values())

    colors = [ORANGE] * len(labels)

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of detections")
    ax.set_title(f"Stage 1 — Detected classes across {summary['total_images']} OPGs "
                 f"(total: {summary['total_detections']})")
    for i, v in enumerate(values):
        ax.text(v + 1, i, str(v), va="center", fontsize=8.5, color=GRAY)
    ax.grid(axis="x", alpha=0.3, linewidth=0.6)

    fig.suptitle(
        "Fig. 4. Frequency of detections by class on the private 50-image OPG dataset (Stage 1).",
        fontsize=10, y=0.02, color=GRAY)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIGS / "fig4_detection_frequency.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_detection_frequency.png")


# ── Fig 5 — Spontaneous recall distribution ───────────────────────────────────
def fig_spontaneous_recall():
    s2_dir = ROOT / "results/stage2_validations"
    if not s2_dir.exists():
        print("  fig5 skipped — no stage2 validations")
        return

    scores = []
    for f in sorted(s2_dir.glob("validation_*.json")):
        data = json.loads(f.read_text())
        sr = data.get("spontaneous_recall")
        if sr is not None:
            scores.append(sr)

    if not scores:
        print("  fig5 skipped — no recall scores")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: histogram
    ax = axes[0]
    bins = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.001]
    n_vals, _, patches = ax.hist(scores, bins=bins,
                                 color=PURPLE, edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(scores), color=RED, linestyle="--", lw=1.8,
               label=f"Mean: {np.mean(scores):.3f}")
    ax.axvline(np.median(scores), color=ORANGE, linestyle=":", lw=1.5,
               label=f"Median: {np.median(scores):.3f}")
    ax.set_xlabel("Spontaneous Recall Rate")
    ax.set_ylabel("Number of images")
    ax.set_title("SR distribution (n = {})".format(len(scores)))
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    # Right: stacked bar with categories
    ax2 = axes[1]
    n_high = sum(1 for s in scores if s >= 0.8)
    n_mid  = sum(1 for s in scores if 0.5 <= s < 0.8)
    n_low  = sum(1 for s in scores if 0.0 < s < 0.5)
    n_zero = sum(1 for s in scores if s == 0.0)
    cats  = ["SR = 0\n(false negative)", "0 < SR < 0.5\n(low)", "0.5–0.8\n(mid)", "SR ≥ 0.8\n(high)"]
    vals  = [n_zero, n_low, n_mid, n_high]
    clrs  = [RED, ORANGE, "#FDD835", GREEN]
    y_pos = np.arange(len(cats))
    ax2.barh(y_pos, vals, color=clrs, edgecolor="white", height=0.55)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(cats)
    ax2.set_xlabel("Number of images")
    ax2.set_title("Breakdown by recall band")
    for i, v in enumerate(vals):
        ax2.text(v + 0.15, i, str(v), va="center", fontsize=9)
    ax2.grid(axis="x", alpha=0.3, linewidth=0.6)

    fig.suptitle(
        "Fig. 5. Spontaneous recall rate distribution across 50 OPGs (Stage 2, n=30 testable images).",
        fontsize=10, y=0.01, color=GRAY)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIGS / "fig5_spontaneous_recall.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig5_spontaneous_recall.png")


# ── Fig 6 — NLP metrics ───────────────────────────────────────────────────────
def fig_nlp_metrics():
    # Read all three metrics from all_metrics.json (single source of truth,
    # same values reported in Table III of the paper).
    metrics_f = ROOT / "results/metrics/all_metrics.json"
    if not metrics_f.exists():
        print("  fig6 skipped — all_metrics.json not found")
        return

    m      = json.loads(metrics_f.read_text()).get("stage3", {})
    bleu_mean  = m.get("bleu4_mean");       bleu_std  = m.get("bleu4_std")
    rouge_mean = m.get("rouge_l_mean");     rouge_std = m.get("rouge_l_std")
    bert_mean  = m.get("bertscore_f1_mean"); bert_std = m.get("bertscore_f1_std")

    if None in (bleu_mean, rouge_mean, bert_mean):
        print("  fig6 skipped — incomplete stage3 metrics in all_metrics.json")
        return

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))

    def _bar(ax, mean_val, std_val, label, color):
        ax.bar([0], [mean_val], width=0.5, color=color, alpha=0.8,
               edgecolor="white", yerr=[[0], [std_val or 0]], capsize=5)
        top = mean_val + (std_val or 0)
        offset = top * 0.25 if top > 0 else 0.001
        ax.text(0, top + offset,
                f"μ={mean_val:.4f}\nσ={std_val:.4f}",
                ha="center", fontsize=8, color=RED)
        ax.set_ylim(0, (top + offset) * 1.35)
        ax.set_xticks([])
        ax.set_ylabel("Score")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3, linewidth=0.6)

    _bar(axes[0], bleu_mean,  bleu_std,  "BLEU-4",       BLUE)
    _bar(axes[1], rouge_mean, rouge_std, "ROUGE-L",      PURPLE)
    _bar(axes[2], bert_mean,  bert_std,  "BERTScore F1", GREEN)

    fig.suptitle(
        "Fig. 6. NLP evaluation of 50 generated pre-clinical reports vs. dentist-written references\n"
        "(Stage 3, Gemini 2.5 Flash, no reference text in LLM prompt).",
        fontsize=10, y=0.01, color=GRAY)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIGS / "fig6_nlp_metrics.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig6_nlp_metrics.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...")
    fig_pipeline()
    fig_training_curves()
    fig_model_comparison()
    fig_detection_frequency()
    fig_spontaneous_recall()
    try:
        fig_nlp_metrics()
    except Exception as e:
        print(f"  fig6 error: {e}")
    print(f"\nAll figures saved to {FIGS}/")
