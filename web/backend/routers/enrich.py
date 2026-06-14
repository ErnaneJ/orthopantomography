from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from database import get_db
from models import Analysis, Detection, Report, UserAnnotation
from schemas import EnrichRequest

STORAGE = Path(__file__).parents[1] / "storage"
router = APIRouter(prefix="/analyses", tags=["enrich"])


def _run_enrichment(analysis_id: int, image_path: Path,
                    context_notes: str):
    """Rebuild annotated image and generate V2 report from existing detections + user regions.
    Does NOT run Stage 2 detection — user-drawn regions stay exactly as drawn."""
    from database import SessionLocal
    from services.report_service import generate as gen_report
    from services.image_service import render_annotated

    db = SessionLocal()
    try:
        analysis = db.get(Analysis, analysis_id)
        analysis.status = "enriching"
        db.commit()

        # Collect existing detections (no new auto-detection)
        all_dets = db.query(Detection).filter_by(analysis_id=analysis_id).all()
        det_dicts = [{"box_x1": d.box_x1, "box_y1": d.box_y1,
                      "box_x2": d.box_x2, "box_y2": d.box_y2,
                      "class_name": d.class_name, "score": d.score,
                      "category": d.category, "is_valid": d.is_valid} for d in all_dets]
        user_regions = [
            {"box_x1": a.box_x1, "box_y1": a.box_y1,
             "box_x2": a.box_x2, "box_y2": a.box_y2,
             "class_name": a.class_name}
            for a in db.query(UserAnnotation)
            .filter_by(analysis_id=analysis_id, kind="region").all()
            if a.box_x1 is not None
        ]

        # Rebuild annotated image with user regions included
        annotated_path = STORAGE / "annotated" / (image_path.stem + "_annotated.jpg")
        render_annotated(image_path, det_dicts, user_regions, annotated_path)

        # V2 report via LLM
        report_data = gen_report(image_path, det_dicts, context_notes, user_regions)
        existing = db.query(Report).filter_by(analysis_id=analysis_id, version=2).first()
        if existing:
            existing.content = report_data["content"]
            existing.model = report_data["model"]
            existing.input_tokens = report_data["input_tokens"]
            existing.output_tokens = report_data["output_tokens"]
        else:
            db.add(Report(analysis_id=analysis_id, version=2, **report_data))
        analysis.status = "done"
        db.commit()
    except Exception as e:
        db.rollback()
        a = db.get(Analysis, analysis_id)
        if a:
            a.status = "error"; a.error_msg = str(e); db.commit()
    finally:
        db.close()


@router.post("/{analysis_id}/enrich")
def enrich(analysis_id: int, body: EnrichRequest, background_tasks: BackgroundTasks,
           request: Request, db: Session = Depends(get_db)):
    a = db.get(Analysis, analysis_id)
    if not a:
        raise HTTPException(404)
    image_path = Path(a.image_path)
    background_tasks.add_task(_run_enrichment, analysis_id, image_path,
                              body.context_notes)
    return {"status": "enriching"}
