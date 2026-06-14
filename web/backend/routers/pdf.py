from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session
from database import get_db
from models import Analysis
from services import pdf_service

router = APIRouter(prefix="/analyses", tags=["pdf"])


@router.get("/{analysis_id}/pdf")
def download_pdf(analysis_id: int, db: Session = Depends(get_db)):
    a = db.get(Analysis, analysis_id)
    if not a:
        raise HTTPException(404)

    det_dicts = [
        {"class_name": d.class_name, "score": d.score, "category": d.category,
         "source": d.source, "is_valid": d.is_valid,
         "box_x1": d.box_x1, "box_y1": d.box_y1,
         "box_x2": d.box_x2, "box_y2": d.box_y2}
        for d in a.detections
    ]
    report_dicts = [
        {"version": r.version, "content": r.content}
        for r in sorted(a.reports, key=lambda r: r.version)
    ]
    ann_dicts = [
        {"kind": ann.kind, "content": ann.content, "class_name": ann.class_name}
        for ann in a.user_annotations
    ]

    pdf_bytes = pdf_service.generate(
        analysis_id=a.id, filename=a.filename,
        image_path=Path(a.image_path),
        annotated_path=Path(a.annotated_path) if a.annotated_path else None,
        detections=det_dicts, reports=report_dicts, annotations=ann_dicts,
    )

    safe_name = a.filename.replace(" ", "_").rsplit(".", 1)[0]
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="opg_{safe_name}_{a.id}.pdf"'},
    )
