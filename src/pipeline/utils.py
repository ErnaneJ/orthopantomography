import re
from pathlib import Path
from config import DESCRIPTIONS_DIR, CLASS_NAMES


def load_description(image_id: str) -> str:
    """Load clinical description for a given image ID (e.g. '00', '01')."""
    path = DESCRIPTIONS_DIR / f"Description_{image_id}.txt"
    if not path.exists():
        return ""
    return path.read_text().strip()


def extract_findings_from_text(text: str) -> list[str]:
    """
    Parse a clinical description into a list of individual finding strings.
    Splits on numbered list patterns like '1.', '2.', etc.
    """
    parts = re.split(r'\d+\.\s*', text)
    return [p.strip() for p in parts if p.strip()]


def match_findings_to_classes(findings: list[str]) -> list[str]:
    """
    Given parsed finding strings, return which CLASS_NAMES are mentioned.
    Simple keyword matching — good enough for coverage scoring.
    """
    mentioned = []
    lower_findings = " ".join(findings).lower()
    aliases = {
        "caries": "Caries", "carious": "Caries",
        "impacted": "Impacted tooth", "impaction": "Impacted tooth",
        "bone loss": "Bone Loss",
        "periapical": "Periapical lesion",
        "root canal": "Root Canal Treatment",
        "crown": "Crown",
        "filling": "Filling",
        "implant": "Implant",
        "missing": "Missing teeth",
        "retained root": "Retained root",
        "septic root": "Retained root",
        "root piece": "Root Piece",
        "fracture": "Fractured teeth", "fractured": "Fractured teeth",
        "cyst": "Cyst",
        "resorption": "Root resorption",
        "attrition": "Attrition",
        "bone defect": "Bone defect",
        "supra": "Supra Eruption",
        "malaligned": "Malaligned",
        "post": "Post-core",
        "plating": "Plating",
        "bracket": "Orthodontic brackets",
        "wire": "Wire",
        "retainer": "Permanent retainer",
        "implant": "Implant",
        "abutment": "Abutment",
    }
    for keyword, cls in aliases.items():
        if keyword in lower_findings and cls not in mentioned:
            mentioned.append(cls)
    return mentioned


def get_all_image_ids() -> list[str]:
    from config import IMAGES_DIR
    ids = sorted([p.stem.replace("Image_", "") for p in IMAGES_DIR.glob("Image_*.jpg")])
    return ids
