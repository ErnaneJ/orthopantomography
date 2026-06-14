from pathlib import Path

ROOT = Path(__file__).parents[2]
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "Images"
DESCRIPTIONS_DIR = DATA_DIR / "Descriptions"
RESULTS_DIR = ROOT / "results"

CLASS_NAMES = [
    "Caries",               # 0
    "Crown",                # 1
    "Filling",              # 2
    "Implant",              # 3
    "Malaligned",           # 4
    "Mandibular Canal",     # 5
    "Missing teeth",        # 6
    "Periapical lesion",    # 7
    "Retained root",        # 8
    "Root Canal Treatment", # 9
    "Root Piece",           # 10
    "Impacted tooth",       # 11
    "Maxillary sinus",      # 12
    "Bone Loss",            # 13
    "Fractured teeth",      # 14
    "Permanent Teeth",      # 15
    "Supra Eruption",       # 16
    "TAD",                  # 17
    "Abutment",             # 18
    "Attrition",            # 19
    "Bone defect",          # 20
    "Gingival former",      # 21
    "Metal band",           # 22
    "Orthodontic brackets", # 23
    "Permanent retainer",   # 24
    "Post-core",            # 25
    "Plating",              # 26
    "Wire",                 # 27
    "Cyst",                 # 28
    "Root resorption",      # 29
    "Primary teeth",        # 30
]

CLASS_TO_CATEGORY = {
    0: "Diseases", 7: "Diseases", 13: "Diseases", 14: "Diseases",
    28: "Diseases", 29: "Diseases", 20: "Diseases", 19: "Diseases",
    1: "Treatments", 2: "Treatments", 3: "Treatments", 9: "Treatments",
    25: "Treatments", 26: "Treatments", 18: "Treatments", 21: "Treatments",
    6: "Tooth Status", 11: "Tooth Status", 8: "Tooth Status",
    10: "Tooth Status", 15: "Tooth Status", 16: "Tooth Status", 4: "Tooth Status",
    5: "Anatomy", 12: "Anatomy", 30: "Anatomy",
    23: "Orthodontics", 27: "Orthodontics", 24: "Orthodontics",
    17: "Orthodontics", 22: "Orthodontics",
}

CATEGORY_COLORS = {
    "Diseases":     (220, 20,  60),
    "Treatments":   (30,  144, 255),
    "Tooth Status": (50,  205, 50),
    "Anatomy":      (255, 165, 0),
    "Orthodontics": (148, 0,   211),
}

# Grounding DINO — text prompt para o modelo (ponto final obrigatório)
DETECTION_PROMPT = ". ".join(CLASS_NAMES) + "."

# Thresholds
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20

# Grounding DINO (kept as zero-shot baseline)
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# YOLOv11 fine-tuned on DentexChallenge 2023 — primary detection model
YOLO_MODEL_PATH = ROOT / "models" / "yolo11_dentex.pt"
YOLO_CONF = 0.25
YOLO_IOU  = 0.45
YOLO_CLASS_TO_CATEGORY = {
    "Caries":            "Diseases",
    "Periapical lesion": "Diseases",
    "Impacted tooth":    "Tooth Status",
}
