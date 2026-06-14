import sys, os
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

STORAGE = Path(__file__).parent / "storage"
STORAGE.joinpath("uploads").mkdir(parents=True, exist_ok=True)
STORAGE.joinpath("annotated").mkdir(parents=True, exist_ok=True)

YOLO_MODEL = Path(__file__).parents[2] / "models" / "yolo11_dentex.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if YOLO_MODEL.exists():
        from ultralytics import YOLO
        print(f"⏳ Loading YOLOv11 ({YOLO_MODEL.name}) ...")
        app.state.model    = YOLO(str(YOLO_MODEL))
        app.state.model_type = "yolo"
        print("✅ YOLOv11 ready.")
    else:
        # Fallback to Grounding DINO if YOLO model not yet trained
        print(f"⚠️  YOLOv11 model not found at {YOLO_MODEL}")
        print("   Falling back to Grounding DINO (zero-shot).")
        print("   Train first: python src/training/train_yolo.py")
        sys.path.insert(0, str(Path(__file__).parents[2] / "src" / "pipeline"))
        from stage1_detection import load_model
        print("⏳ Loading Grounding DINO ... (15–20s)")
        app.state.processor, app.state.model, app.state.device = load_model()
        app.state.model_type = "gdino"
        print("✅ Grounding DINO ready.")

    from database import engine
    from models import Base
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="OPG Analysis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"], allow_headers=["*"],
)

app.mount("/api/static/uploads",
          StaticFiles(directory=STORAGE / "uploads"), name="uploads")
app.mount("/api/static/annotated",
          StaticFiles(directory=STORAGE / "annotated"), name="annotated")

from routers import analyses, detections, annotations, enrich, pdf  # noqa
app.include_router(analyses.router, prefix="/api")
app.include_router(detections.router, prefix="/api")
app.include_router(annotations.router, prefix="/api")
app.include_router(enrich.router, prefix="/api")
app.include_router(pdf.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"ok": True, "model": getattr(app.state, "model_type", "unknown")}
