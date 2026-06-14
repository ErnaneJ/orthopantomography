import sys
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "pipeline"))
from config import CATEGORY_COLORS


def render_annotated(image_path: Path, detections: list[dict],
                     user_regions: list[dict], out_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    for det in detections:
        if not det.get("is_valid", 1):
            continue
        x1, y1, x2, y2 = det["box_x1"], det["box_y1"], det["box_x2"], det["box_y2"]
        color = CATEGORY_COLORS.get(det["category"], (128, 128, 128))
        draw.rectangle([x1, y1, x2, y2], fill=color + (40,), outline=color + (220,), width=3)
        label = f"{det['class_name']} {det['score']:.0%}"
        tw = len(label) * 8
        draw.rectangle([x1, y1 - 20, x1 + tw, y1], fill=color + (200,))
        draw.text((x1 + 2, y1 - 17), label, fill="white")

    # user-drawn regions in yellow
    for reg in user_regions:
        x1, y1 = reg["box_x1"], reg["box_y1"]
        x2, y2 = reg["box_x2"], reg["box_y2"]
        label = reg.get("class_name") or "User"
        draw.rectangle([x1, y1, x2, y2], fill=(255, 215, 0, 50),
                       outline=(255, 215, 0, 230), width=3)
        tw = len(label) * 8
        draw.rectangle([x1, y1 - 20, x1 + tw, y1], fill=(255, 215, 0, 200))
        draw.text((x1 + 2, y1 - 17), label, fill="black")

    img.save(out_path, quality=90)
