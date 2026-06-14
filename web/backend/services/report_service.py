import os, base64, sys
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "pipeline"))

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL = "google/gemini-2.5-flash"

SYSTEM = """You are an expert dental radiologist analyzing panoramic dental radiographs (OPGs).
Generate a structured pre-clinical dental report using FDI tooth notation (11–18, 21–28, 31–38, 41–48).

Format EXACTLY as:

## PANORAMIC RADIOGRAPH PRE-CLINICAL REPORT

### Overall Assessment
[1-2 sentences]

### Tooth-by-Tooth Analysis
**Tooth [FDI]:** [finding or "Within normal limits"]

### Detected Pathologies Summary
- [Pathology — location — severity]

### Recommended Clinical Actions
- [Recommendation]

### Radiographic Quality Notes
[Quality notes]"""

TEMPLATE = """Analyze this OPG and generate a structured pre-clinical report.

Automated detection findings:
{detections_summary}

{enrichment_block}Generate the complete structured report."""

ENRICHMENT_BLOCK = """Additional clinical context provided by the operator:
{context_notes}

User-identified regions of interest:
{regions_summary}

"""


def _encode(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def _det_summary(detections: list[dict]) -> str:
    valid = [d for d in detections if d.get("is_valid", 1)]
    if not valid:
        return "No pathologies detected automatically."
    return "\n".join(
        f"- {d['class_name']} (conf: {d['score']:.0%}, category: {d['category']})"
        for d in valid[:15]
    )


def generate(image_path: Path, detections: list[dict],
             context_notes: str = "", user_regions: list[dict] | None = None) -> dict:
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE,
    )
    enrichment = ""
    if context_notes or user_regions:
        reg_summary = "\n".join(
            f"- {r.get('class_name','Region')} at ({r['box_x1']:.0f},{r['box_y1']:.0f})"
            f"–({r['box_x2']:.0f},{r['box_y2']:.0f})"
            for r in (user_regions or [])
        ) or "None"
        enrichment = ENRICHMENT_BLOCK.format(
            context_notes=context_notes or "None",
            regions_summary=reg_summary,
        )

    user_text = TEMPLATE.format(
        detections_summary=_det_summary(detections),
        enrichment_block=enrichment,
    )

    resp = client.chat.completions.create(
        model=MODEL, max_tokens=1500,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_encode(image_path)}"}},
                {"type": "text", "text": user_text},
            ]},
        ],
    )
    return {
        "content": resp.choices[0].message.content,
        "model": MODEL,
        "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
        "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
    }
