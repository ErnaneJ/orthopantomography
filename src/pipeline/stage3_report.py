"""
Stage 3 — Structured pre-clinical report generation via OpenRouter (vision).
Input: OPG image + detections from Stage 1.
Output: tooth-by-tooth structured pre-clinical report.
"""
import base64
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import IMAGES_DIR, RESULTS_DIR
from utils import load_description

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemini-2.5-flash"   # fast + cheap + strong vision

REPORT_SYSTEM = """You are an expert dental radiologist analyzing panoramic dental radiographs (OPGs).
Generate a structured pre-clinical dental report. Use FDI tooth notation (11–18, 21–28, 31–38, 41–48).

Format your response EXACTLY as:

## PANORAMIC RADIOGRAPH PRE-CLINICAL REPORT

### Overall Assessment
[1-2 sentences summarizing the key findings]

### Tooth-by-Tooth Analysis
List only teeth with notable findings:
**Tooth [FDI number]:** [finding description or "Within normal limits"]

### Detected Pathologies Summary
- [Pathology — location — severity if assessable]

### Recommended Clinical Actions
- [Specific recommendation]

### Radiographic Quality Notes
[Image quality, artifacts, or limitations]

Be specific and clinical. Only report what is clearly visible."""

REPORT_USER_TEMPLATE = """Analyze this panoramic dental radiograph (OPG) and generate a structured pre-clinical report.

Automated detection (YOLOv11, fine-tuned on DentexChallenge 2023) identified these findings:
{detections_summary}

Generate the complete structured report based on the image and the detections above.
Do not assume any findings beyond what is visible in the image."""


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def format_detections_summary(detections: list[dict]) -> str:
    if not detections:
        return "No pathologies detected automatically."
    lines = [f"- {d['class']} (conf: {d['score']:.0%}, category: {d['category']})"
             for d in detections[:15]]
    return "\n".join(lines)


def generate_report(client: OpenAI, image_path: Path, img_id: str,
                    stage1_result: dict | None = None) -> dict:
    description = load_description(img_id)
    detections = stage1_result["detections"] if stage1_result else []

    user_content = REPORT_USER_TEMPLATE.format(
        detections_summary=format_detections_summary(detections),
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        },
                    },
                    {"type": "text", "text": user_content},
                ],
            },
        ],
    )

    report_text = response.choices[0].message.content
    usage = response.usage

    return {
        "image": image_path.name,
        "image_id": img_id,
        "model": MODEL,
        "reference_description": description,
        "detections_used": len(detections),
        "report": report_text,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
    }


def run(stage1_results: list[dict] | None = None):
    out_dir = RESULTS_DIR / "stage3_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    image_paths = sorted(IMAGES_DIR.glob("Image_*.jpg"))

    s1_index = {}
    if stage1_results:
        s1_index = {r["image"]: r for r in stage1_results}
    else:
        # load from disk if running standalone
        s1_dir = RESULTS_DIR / "stage1_detections"
        for f in s1_dir.glob("detection_*.json"):
            d = json.loads(f.read_text())
            s1_index[d["image"]] = d

    all_results = []
    total_in, total_out = 0, 0

    for img_path in tqdm(image_paths, desc="Gerando pré-laudos"):
        img_id = img_path.stem.replace("Image_", "")
        s1 = s1_index.get(img_path.name)

        try:
            result = generate_report(client, img_path, img_id, s1)
        except Exception as e:
            result = {
                "image": img_path.name, "image_id": img_id,
                "model": MODEL, "reference_description": load_description(img_id),
                "detections_used": 0, "report": f"ERROR: {e}",
                "input_tokens": 0, "output_tokens": 0,
            }

        all_results.append(result)
        total_in  += result["input_tokens"]
        total_out += result["output_tokens"]

        (out_dir / f"report_{img_id}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False)
        )
        (out_dir / f"report_{img_id}.txt").write_text(result["report"])

        time.sleep(0.3)

    summary = {
        "total_images": len(all_results),
        "model": MODEL,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "errors": sum(1 for r in all_results if r["report"].startswith("ERROR")),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    ok = len(all_results) - summary["errors"]
    print(f"\n✓ Stage 3 | {ok}/50 relatórios gerados | "
          f"tokens: {total_in:,} in / {total_out:,} out")
    return all_results


if __name__ == "__main__":
    run()
