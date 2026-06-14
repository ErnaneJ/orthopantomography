import sys
import warnings
from pathlib import Path

import torch
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "pipeline"))
from config import BOX_THRESHOLD, TEXT_THRESHOLD, GDINO_MODEL_ID
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image


def run_stage2(processor, model, device, image_path: Path,
               class_names: list[str]) -> list[dict]:
    if not class_names:
        return []
    text_labels = [[c.lower() for c in class_names]]
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
        text_labels=text_labels,
    )[0]
    return [
        {"box_x1": float(b[0]), "box_y1": float(b[1]),
         "box_x2": float(b[2]), "box_y2": float(b[3]),
         "class_name": str(lb), "score": float(sc),
         "category": "User", "source": "user", "is_valid": 1}
        for sc, lb, b in zip(results["scores"], results["text_labels"], results["boxes"])
    ]
