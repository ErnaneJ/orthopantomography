"""
Stage 4 — Compute NLP metrics comparing generated reports vs reference descriptions.
Metrics: BLEU-4, ROUGE-L, BERTScore (for Stage 3).
Also computes detection statistics (Stage 1) and coverage statistics (Stage 2).
"""
import json
import sys
from pathlib import Path

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, CLASS_NAMES, CATEGORY_COLORS

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    if not hyp_tokens:
        return 0.0
    sf = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=sf)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def run():
    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = RESULTS_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1 statistics ──────────────────────────────────────────────────
    s1_dir = RESULTS_DIR / "stage1_detections"
    s1_summary = json.loads((s1_dir / "summary.json").read_text()) if (s1_dir / "summary.json").exists() else {}

    # per-image detection counts
    det_counts = []
    class_freq = {}
    for f in sorted(s1_dir.glob("detection_*.json")):
        data = json.loads(f.read_text())
        det_counts.append(data["count"])
        for d in data["detections"]:
            class_freq[d["class"]] = class_freq.get(d["class"], 0) + 1

    # ── Stage 2 statistics (Spontaneous Recall — unbiased) ──────────────────
    s2_dir = RESULTS_DIR / "stage2_validations"
    coverage_scores = []   # now = spontaneous_recall scores
    for f in sorted(s2_dir.glob("validation_*.json")):
        data = json.loads(f.read_text())
        # Support both old (coverage_score) and new (spontaneous_recall) field names
        score = data.get("spontaneous_recall") or data.get("coverage_score")
        if score is not None:
            coverage_scores.append(score)

    # ── Stage 3 NLP metrics ─────────────────────────────────────────────────
    s3_dir = RESULTS_DIR / "stage3_reports"
    bleu_scores, rouge_scores = [], []

    for f in sorted(s3_dir.glob("report_*.json")):
        data = json.loads(f.read_text())
        ref = data.get("reference_description", "")
        hyp = data.get("report", "")
        if ref and hyp:
            bleu_scores.append(compute_bleu(ref, hyp))
            rouge_scores.append(compute_rouge_l(ref, hyp))

    # ── BERTScore (optional — loads model, ~2min first run) ─────────────────
    bert_f1_scores = []
    try:
        from bert_score import score as bert_score
        refs, hyps = [], []
        for f in sorted(s3_dir.glob("report_*.json")):
            data = json.loads(f.read_text())
            if data.get("reference_description") and data.get("report"):
                refs.append(data["reference_description"])
                hyps.append(data["report"])
        if refs:
            print("Computando BERTScore (pode levar 1-2 min)...")
            _, _, F1 = bert_score(hyps, refs, lang="en", verbose=False)
            bert_f1_scores = F1.tolist()
    except Exception as e:
        print(f"BERTScore pulado: {e}")

    # ── mAP from DentexChallenge evaluation (if available) ──────────────────
    eval_file = RESULTS_DIR / "evaluation" / "dentex_test_metrics.json"
    dentex_metrics = {}
    if eval_file.exists():
        dentex_metrics = json.loads(eval_file.read_text())

    comparison_file = RESULTS_DIR / "evaluation" / "model_comparison.json"
    comparison_metrics = {}
    if comparison_file.exists():
        comparison_metrics = json.loads(comparison_file.read_text())

    # ── Compile final metrics ────────────────────────────────────────────────
    metrics = {
        "stage1": {
            "model": s1_summary.get("model", "YOLOv11 (fine-tuned)"),
            "total_images": len(det_counts),
            "total_detections": sum(det_counts),
            "avg_per_image": round(np.mean(det_counts), 2) if det_counts else 0,
            "std_per_image": round(np.std(det_counts), 2) if det_counts else 0,
            "min_per_image": int(min(det_counts)) if det_counts else 0,
            "max_per_image": int(max(det_counts)) if det_counts else 0,
            "classes_detected": len(class_freq),
            "top10_classes": dict(sorted(class_freq.items(), key=lambda x: -x[1])[:10]),
        },
        "stage2": {
            "metric": "Spontaneous Recall Rate (unbiased)",
            "images_with_findings": len(coverage_scores),
            "avg_spontaneous_recall": round(np.mean(coverage_scores), 4) if coverage_scores else 0,
            "std_spontaneous_recall": round(np.std(coverage_scores), 4) if coverage_scores else 0,
            "high_recall_pct": round(sum(1 for s in coverage_scores if s >= 0.8) / len(coverage_scores) * 100, 1) if coverage_scores else 0,
        },
        "stage3": {
            "reports_evaluated": len(bleu_scores),
            "bleu4_mean": round(np.mean(bleu_scores), 4) if bleu_scores else 0,
            "bleu4_std": round(np.std(bleu_scores), 4) if bleu_scores else 0,
            "rouge_l_mean": round(np.mean(rouge_scores), 4) if rouge_scores else 0,
            "rouge_l_std": round(np.std(rouge_scores), 4) if rouge_scores else 0,
            "bertscore_f1_mean": round(np.mean(bert_f1_scores), 4) if bert_f1_scores else None,
            "bertscore_f1_std": round(np.std(bert_f1_scores), 4) if bert_f1_scores else None,
            "note": "Evaluated without reference in LLM prompt (no data leakage)",
        },
        "dentex_evaluation": dentex_metrics.get("aggregate", {}),
        "model_comparison": comparison_metrics.get("models", {}),
    }
    (metrics_dir / "all_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── Figures ──────────────────────────────────────────────────────────────
    _plot_class_frequency(class_freq, figs_dir)
    _plot_coverage_distribution(coverage_scores, figs_dir)
    _plot_nlp_metrics(bleu_scores, rouge_scores, bert_f1_scores, figs_dir)

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║              FINAL RESULTS SUMMARY                  ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║ Stage 1 — Detection (YOLOv11 fine-tuned)            ║")
    print(f"║   Avg findings/image : {metrics['stage1']['avg_per_image']:.1f} ± {metrics['stage1']['std_per_image']:.1f}")
    print(f"║   Classes detected   : {metrics['stage1']['classes_detected']}")
    if metrics.get("dentex_evaluation"):
        ev = metrics["dentex_evaluation"]
        print(f"║   mAP@50 (DentexTest): {ev.get('mAP50', 'N/A')}")
        print(f"║   mAP@50:95          : {ev.get('mAP50_95', 'N/A')}")
    print(f"║ Stage 2 — Spontaneous Recall (unbiased)             ║")
    print(f"║   Mean SR  : {metrics['stage2']['avg_spontaneous_recall']:.1%} ± {metrics['stage2']['std_spontaneous_recall']:.1%}")
    print(f"║   High (≥80%): {metrics['stage2']['high_recall_pct']:.0f}% of images")
    print(f"║ Stage 3 — Report Generation (no leakage)            ║")
    print(f"║   BLEU-4   : {metrics['stage3']['bleu4_mean']:.4f} ± {metrics['stage3']['bleu4_std']:.4f}")
    print(f"║   ROUGE-L  : {metrics['stage3']['rouge_l_mean']:.4f} ± {metrics['stage3']['rouge_l_std']:.4f}")
    if bert_f1_scores:
        print(f"║   BERTScore: {metrics['stage3']['bertscore_f1_mean']:.4f} ± {metrics['stage3']['bertscore_f1_std']:.4f}")
    print("╚══════════════════════════════════════════════════════╝")
    return metrics


def _plot_class_frequency(class_freq: dict, figs_dir: Path):
    if not class_freq:
        return
    top_n = dict(sorted(class_freq.items(), key=lambda x: -x[1])[:15])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(list(top_n.keys()), list(top_n.values()), color="#2196F3")
    ax.set_xlabel("Frequency (detections)")
    ax.set_title("Top 15 Detected Classes — Stage 1")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(figs_dir / "stage1_class_frequency.png", dpi=150)
    plt.close(fig)


def _plot_coverage_distribution(coverage_scores: list, figs_dir: Path):
    if not coverage_scores:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(coverage_scores, bins=10, range=(0, 1), color="#4CAF50", edgecolor="white")
    ax.axvline(np.mean(coverage_scores), color="red", linestyle="--",
               label=f"Mean: {np.mean(coverage_scores):.2f}")
    ax.set_xlabel("Coverage Score")
    ax.set_ylabel("Number of Images")
    ax.set_title("Stage 2 — Report Coverage Distribution")
    ax.legend()
    plt.tight_layout()
    fig.savefig(figs_dir / "stage2_coverage_distribution.png", dpi=150)
    plt.close(fig)


def _plot_nlp_metrics(bleu: list, rouge: list, bert: list, figs_dir: Path):
    metrics_data = {"BLEU-4": bleu, "ROUGE-L": rouge}
    if bert:
        metrics_data["BERTScore F1"] = bert
    if not any(metrics_data.values()):
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = range(len(metrics_data))
    means = [np.mean(v) for v in metrics_data.values()]
    stds  = [np.std(v)  for v in metrics_data.values()]
    bars = ax.bar(positions, means, yerr=stds, capsize=5,
                  color=["#FF5722", "#9C27B0", "#00BCD4"][:len(metrics_data)],
                  edgecolor="white", width=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(metrics_data.keys())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Stage 3 — Report Generation NLP Metrics")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{mean:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(figs_dir / "stage3_nlp_metrics.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
