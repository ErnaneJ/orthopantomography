"""
Input ablation for Stage 3 report generation: image-only vs. detections-only
vs. both (the pipeline's default configuration, already evaluated in
results/stage3_reports/).

Tests whether the YOLOv11 detections summary contributes information beyond
the image itself, and whether the image contributes beyond the detections
summary, by holding the LLM, prompt structure, and 50-image set fixed and
varying only what is included in the user message.

Run: python src/evaluation/ablation_llm_input.py
"""
import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[1] / "pipeline"))
from config import IMAGES_DIR, RESULTS_DIR
from utils import load_description

sys.path.insert(0, str(Path(__file__).parents[2]))
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemini-2.5-flash"
OUT_DIR = RESULTS_DIR / "evaluation" / "llm_input_ablation"
S1_DIR = RESULTS_DIR / "stage1_detections"

REPORT_SYSTEM = """You are an expert dental radiologist analyzing panoramic dental radiographs (OPGs).
Generate a structured pre-clinical dental report. Use FDI tooth notation (11-18, 21-28, 31-38, 41-48).

Format your response EXACTLY as:

## PANORAMIC RADIOGRAPH PRE-CLINICAL REPORT

### Overall Assessment
[1-2 sentences summarizing the key findings]

### Tooth-by-Tooth Analysis
List only teeth with notable findings:
**Tooth [FDI number]:** [finding description or "Within normal limits"]

### Detected Pathologies Summary
- [Pathology - location - severity if assessable]

### Recommended Clinical Actions
- [Specific recommendation]

### Radiographic Quality Notes
[Image quality, artifacts, or limitations]

Be specific and clinical. Only report what is clearly visible."""

IMAGE_ONLY_TEMPLATE = """Analyze this panoramic dental radiograph (OPG) and generate a structured pre-clinical report.

No automated detection is available for this case. Base your report solely on visual inspection of the image."""

DETECTIONS_ONLY_TEMPLATE = """Generate a structured pre-clinical report for a panoramic dental radiograph (OPG) based solely on the following automated detections (YOLOv11, fine-tuned on DentexChallenge 2023). No image is available to you.

{detections_summary}

Generate the complete structured report using only the detections above. Do not invent findings beyond what is listed."""


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def format_detections_summary(detections: list[dict]) -> str:
    if not detections:
        return "No pathologies detected automatically."
    lines = [f"- {d['class']} (conf: {d['score']:.0%}, category: {d['category']})"
             for d in detections[:15]]
    return "\n".join(lines)


def generate_image_only(client: OpenAI, image_path: Path) -> str:
    response = client.chat.completions.create(
        model=MODEL, max_tokens=1500,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(image_path)}"}},
                {"type": "text", "text": IMAGE_ONLY_TEMPLATE},
            ]},
        ],
    )
    return response.choices[0].message.content


def generate_detections_only(client: OpenAI, detections: list[dict]) -> str:
    user_text = DETECTIONS_ONLY_TEMPLATE.format(
        detections_summary=format_detections_summary(detections))
    response = client.chat.completions.create(
        model=MODEL, max_tokens=1500,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM},
            {"role": "user", "content": user_text},
        ],
    )
    return response.choices[0].message.content


def compute_bleu(reference: str, hypothesis: str) -> float:
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    if not hyp_tokens:
        return 0.0
    sf = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=sf)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, hypothesis)["rougeL"].fmeasure


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    image_paths = sorted(IMAGES_DIR.glob("Image_*.jpg"))

    s1_index = {}
    for f in S1_DIR.glob("detection_*.json"):
        d = json.loads(f.read_text())
        s1_index[d["image"]] = d

    conditions = {"image_only": [], "detections_only": []}
    refs = {"image_only": [], "detections_only": []}

    for img_path in tqdm(image_paths, desc="Input ablation (image-only + detections-only)"):
        img_id = img_path.stem.replace("Image_", "")
        cache_f = OUT_DIR / f"case_{img_id}.json"
        s1 = s1_index.get(img_path.name, {})
        detections = s1.get("detections", [])
        ref = load_description(img_id)

        if cache_f.exists():
            case = json.loads(cache_f.read_text())
        else:
            case = {"image_id": img_id, "reference_description": ref}
            try:
                case["image_only_report"] = generate_image_only(client, img_path)
            except Exception as e:
                case["image_only_report"] = f"ERROR: {e}"
            time.sleep(0.3)
            try:
                case["detections_only_report"] = generate_detections_only(client, detections)
            except Exception as e:
                case["detections_only_report"] = f"ERROR: {e}"
            time.sleep(0.3)
            cache_f.write_text(json.dumps(case, indent=2, ensure_ascii=False))

        if ref and not case["image_only_report"].startswith("ERROR"):
            conditions["image_only"].append(case["image_only_report"])
            refs["image_only"].append(ref)
        if ref and not case["detections_only_report"].startswith("ERROR"):
            conditions["detections_only"].append(case["detections_only_report"])
            refs["detections_only"].append(ref)

    # ── "Both" condition: reuse the existing full-pipeline reports (no re-run) ──
    s3_dir = RESULTS_DIR / "stage3_reports"
    both_hyps, both_refs = [], []
    for f in sorted(s3_dir.glob("report_*.json")):
        d = json.loads(f.read_text())
        if d.get("reference_description") and d.get("report") and not d["report"].startswith("ERROR"):
            both_hyps.append(d["report"])
            both_refs.append(d["reference_description"])

    results = {}
    for name, hyps, rs in [
        ("image_only", conditions["image_only"], refs["image_only"]),
        ("detections_only", conditions["detections_only"], refs["detections_only"]),
        ("both", both_hyps, both_refs),
    ]:
        bleu = [compute_bleu(r, h) for r, h in zip(rs, hyps)]
        rouge = [compute_rouge_l(r, h) for r, h in zip(rs, hyps)]
        _, _, F1 = bert_score(hyps, rs, lang="en", verbose=False)
        _, _, F1_resc = bert_score(hyps, rs, lang="en", rescale_with_baseline=True, verbose=False)
        results[name] = {
            "n": len(hyps),
            "bleu4_mean": round(float(np.mean(bleu)), 4),
            "rouge_l_mean": round(float(np.mean(rouge)), 4),
            "bertscore_f1_mean": round(float(F1.mean()), 4),
            "bertscore_f1_std": round(float(F1.std()), 4),
            "bertscore_f1_rescaled_mean": round(float(F1_resc.mean()), 4),
        }

    out_file = RESULTS_DIR / "evaluation" / "llm_input_ablation_summary.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()
