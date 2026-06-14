from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import Analysis, UserAnnotation
from schemas import AnnotationCreate, AnnotationOut

router = APIRouter(prefix="/analyses", tags=["annotations"])


@router.post("/{analysis_id}/annotations", response_model=AnnotationOut)
def create_annotation(analysis_id: int, body: AnnotationCreate,
                      db: Session = Depends(get_db)):
    if not db.get(Analysis, analysis_id):
        raise HTTPException(404)
    ann = UserAnnotation(
        analysis_id=analysis_id, kind=body.kind, content=body.content,
        class_name=body.class_name,
        box_x1=body.box_x1, box_y1=body.box_y1,
        box_x2=body.box_x2, box_y2=body.box_y2,
    )
    db.add(ann)
    db.commit()
    db.refresh(ann)
    return AnnotationOut.model_validate(ann)


@router.delete("/{analysis_id}/annotations/{ann_id}")
def delete_annotation(analysis_id: int, ann_id: int, db: Session = Depends(get_db)):
    ann = db.get(UserAnnotation, ann_id)
    if not ann or ann.analysis_id != analysis_id:
        raise HTTPException(404)
    db.delete(ann)
    db.commit()
    return {"deleted": True}
