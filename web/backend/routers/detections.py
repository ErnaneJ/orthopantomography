from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import get_db
from models import Detection, Analysis
from services.image_service import render_annotated
from pathlib import Path

STORAGE = Path(__file__).parents[1] / "storage"
router = APIRouter(prefix="/detections", tags=["detections"])


@router.delete("/{detection_id}")
def delete_detection(detection_id: int, db: Session = Depends(get_db)):
    det = db.get(Detection, detection_id)
    if not det:
        raise HTTPException(404)
    analysis_id = det.analysis_id
    db.delete(det)
    db.commit()
    # Rebuild annotated image without deleted detection
    _rebuild_annotated(analysis_id, db)
    return {"deleted": True, "id": detection_id}


class DetectClassRequest(BaseModel):
    class_name: str


@router.post("/analyses/{analysis_id}/detect-class")
def detect_class(analysis_id: int, body: DetectClassRequest,
                 request: Request, db: Session = Depends(get_db)):
    """Immediately run Stage 2 for a single class and save detections."""
    a = db.get(Analysis, analysis_id)
    if not a:
        raise HTTPException(404)
    from services.validation_service import run_stage2
    new_dets = run_stage2(
        request.app.state.processor, request.app.state.model,
        request.app.state.device, Path(a.image_path), [body.class_name]
    )
    saved = []
    for d in new_dets:
        det = Detection(
            analysis_id=analysis_id, class_name=d["class_name"],
            score=d["score"], box_x1=d["box_x1"], box_y1=d["box_y1"],
            box_x2=d["box_x2"], box_y2=d["box_y2"],
            category=d["category"], source="user",
        )
        db.add(det)
        saved.append(det)
    db.commit()
    for s in saved:
        db.refresh(s)
    _rebuild_annotated(analysis_id, db)
    return {"detected": len(saved), "class_name": body.class_name}


def _rebuild_annotated(analysis_id: int, db: Session):
    a = db.get(Analysis, analysis_id)
    if not a or not a.annotated_path:
        return
    all_dets = db.query(Detection).filter_by(analysis_id=analysis_id).all()
    det_dicts = [
        {"box_x1": d.box_x1, "box_y1": d.box_y1, "box_x2": d.box_x2, "box_y2": d.box_y2,
         "class_name": d.class_name, "score": d.score, "category": d.category, "is_valid": 1}
        for d in all_dets
    ]
    from models import UserAnnotation
    regions = [
        {"box_x1": x.box_x1, "box_y1": x.box_y1, "box_x2": x.box_x2, "box_y2": x.box_y2, "class_name": x.class_name}
        for x in db.query(UserAnnotation).filter_by(analysis_id=analysis_id, kind="region").all()
        if x.box_x1 is not None
    ]
    render_annotated(Path(a.image_path), det_dicts, regions, Path(a.annotated_path))
