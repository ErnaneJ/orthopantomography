import shutil, uuid
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import func as sqla_func
from sqlalchemy.orm import Session

from database import get_db
from models import Analysis, Detection, Report, UserAnnotation
from schemas import AnalysisDetail, AnalysisListItem, DetectionOut, ReportOut, AnnotationOut

STORAGE = Path(__file__).parents[1] / "storage"
router = APIRouter(prefix="/analyses", tags=["analyses"])


def _image_url(req: Request, path: str) -> str:
    fname = Path(path).name
    return str(req.base_url) + f"api/static/uploads/{fname}"


def _annotated_url(req: Request, path: str | None) -> str | None:
    if not path:
        return None
    return str(req.base_url) + f"api/static/annotated/{Path(path).name}"


def _run_pipeline(analysis_id: int, image_path: Path, app_state):
    from database import SessionLocal
    from services.detection_service import run_stage1
    from services.report_service import generate as gen_report
    from services.image_service import render_annotated

    db = SessionLocal()
    try:
        analysis = db.get(Analysis, analysis_id)
        analysis.status = "processing"
        db.commit()

        annotated_path = STORAGE / "annotated" / (image_path.stem + "_annotated.jpg")

        # Stage 1 — YOLO (primary) or Grounding DINO (fallback)
        model_type = getattr(app_state, "model_type", "gdino")
        if model_type == "yolo":
            det_dicts = run_stage1(app_state.model, image_path, annotated_path,
                                   model_type="yolo")
        else:
            det_dicts = run_stage1(app_state.model, image_path, annotated_path,
                                   model_type="gdino",
                                   processor=app_state.processor,
                                   device=app_state.device)
        for d in det_dicts:
            db.add(Detection(
                analysis_id=analysis_id,
                class_name=d["class_name"], raw_label=d.get("raw_label", ""),
                score=d["score"], box_x1=d["box_x1"], box_y1=d["box_y1"],
                box_x2=d["box_x2"], box_y2=d["box_y2"],
                category=d["category"], source="auto",
            ))
        analysis.annotated_path = str(annotated_path)
        db.commit()

        # Stage 3
        report_data = gen_report(image_path, det_dicts)
        db.add(Report(
            analysis_id=analysis_id, version=1,
            content=report_data["content"], model=report_data["model"],
            input_tokens=report_data["input_tokens"],
            output_tokens=report_data["output_tokens"],
        ))
        analysis.status = "done"
        db.commit()
    except Exception as e:
        db.rollback()
        analysis = db.get(Analysis, analysis_id)
        if analysis:
            analysis.status = "error"
            analysis.error_msg = str(e)
            db.commit()
    finally:
        db.close()


@router.post("/upload")
async def upload(file: UploadFile, background_tasks: BackgroundTasks,
                 request: Request, db: Session = Depends(get_db)):
    uid = uuid.uuid4().hex
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    dest = STORAGE / "uploads" / f"{uid}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    analysis = Analysis(filename=file.filename or "upload", image_path=str(dest))
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    background_tasks.add_task(_run_pipeline, analysis.id, dest, request.app.state)
    return {"id": analysis.id, "status": "processing"}


@router.get("", response_model=list[AnalysisListItem])
def list_analyses(request: Request, db: Session = Depends(get_db)):
    rows = db.query(Analysis).order_by(Analysis.created_at.desc()).all()
    return [
        AnalysisListItem(
            id=r.id, filename=r.filename, status=r.status,
            created_at=r.created_at,
            image_url=_image_url(request, r.image_path),
        )
        for r in rows
    ]


@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Aggregate statistics across all analyses — for metrics dashboard and article."""
    total_analyses = db.query(Analysis).count()
    done_analyses  = db.query(Analysis).filter(Analysis.status.in_(["done", "enriching"])).count()

    base_q = db.query(Detection).filter(Detection.is_valid == 1)
    total_detections = base_q.count()

    if total_detections == 0:
        return {
            "total_analyses": total_analyses, "done_analyses": done_analyses,
            "total_detections": 0, "class_stats": [], "category_stats": [],
            "source_stats": [], "confidence_percentiles": {},
            "confidence_histogram": [{"bin": f"{i*10}-{(i+1)*10}%", "count": 0} for i in range(10)],
            "findings_per_analysis": [],
            "findings_summary": {"mean": 0, "min": 0, "max": 0, "std": 0, "total_analyses_with_findings": 0},
        }

    class_stats_raw = (
        db.query(
            Detection.class_name, Detection.category,
            sqla_func.count(Detection.id).label("count"),
            sqla_func.avg(Detection.score).label("avg_conf"),
            sqla_func.min(Detection.score).label("min_conf"),
            sqla_func.max(Detection.score).label("max_conf"),
        ).filter(Detection.is_valid == 1).group_by(Detection.class_name, Detection.category).all()
    )

    prevalence_map = {
        r[0]: r[1] for r in
        db.query(Detection.class_name, sqla_func.count(sqla_func.distinct(Detection.analysis_id)))
        .filter(Detection.is_valid == 1).group_by(Detection.class_name).all()
    }

    class_stats = sorted([{
        "class_name": r.class_name, "category": r.category, "count": r.count,
        "prevalence": prevalence_map.get(r.class_name, 0),
        "avg_confidence": round(float(r.avg_conf), 4),
        "min_confidence": round(float(r.min_conf), 4),
        "max_confidence": round(float(r.max_conf), 4),
    } for r in class_stats_raw], key=lambda x: x["count"], reverse=True)

    category_stats = [
        {"category": r[0], "count": r[1]}
        for r in db.query(Detection.category, sqla_func.count(Detection.id))
        .filter(Detection.is_valid == 1).group_by(Detection.category)
        .order_by(sqla_func.count(Detection.id).desc()).all()
    ]
    source_stats = [
        {"source": r[0], "count": r[1]}
        for r in db.query(Detection.source, sqla_func.count(Detection.id))
        .filter(Detection.is_valid == 1).group_by(Detection.source).all()
    ]

    counts = [
        r[1] for r in
        db.query(Detection.analysis_id, sqla_func.count(Detection.id))
        .filter(Detection.is_valid == 1).group_by(Detection.analysis_id).all()
    ]
    mean_c = sum(counts) / len(counts)
    std_c  = (sum((c - mean_c) ** 2 for c in counts) / len(counts)) ** 0.5

    all_scores = sorted(r[0] for r in db.query(Detection.score).filter(Detection.is_valid == 1).all())
    n = len(all_scores)
    bins = [0] * 10
    for s in all_scores:
        bins[min(int(s * 10), 9)] += 1

    return {
        "total_analyses": total_analyses, "done_analyses": done_analyses,
        "total_detections": total_detections,
        "class_stats": class_stats, "category_stats": category_stats, "source_stats": source_stats,
        "confidence_percentiles": {
            "p25": round(all_scores[int(n * 0.25)], 4),
            "p50": round(all_scores[int(n * 0.50)], 4),
            "p75": round(all_scores[int(n * 0.75)], 4),
            "p90": round(all_scores[int(n * 0.90)], 4),
            "mean": round(sum(all_scores) / n, 4),
        },
        "confidence_histogram": [{"bin": f"{i*10}-{(i+1)*10}%", "count": bins[i]} for i in range(10)],
        "findings_per_analysis": counts,
        "findings_summary": {
            "mean": round(mean_c, 2), "min": min(counts), "max": max(counts),
            "std": round(std_c, 2), "total_analyses_with_findings": len(counts),
        },
    }


@router.get("/{analysis_id}", response_model=AnalysisDetail)
def get_analysis(analysis_id: int, request: Request, db: Session = Depends(get_db)):
    a = db.get(Analysis, analysis_id)
    if not a:
        raise HTTPException(404)
    return AnalysisDetail(
        id=a.id, filename=a.filename, status=a.status,
        error_msg=a.error_msg, created_at=a.created_at,
        image_url=_image_url(request, a.image_path),
        annotated_url=_annotated_url(request, a.annotated_path),
        detections=[DetectionOut.model_validate(d) for d in a.detections],
        reports=[ReportOut.model_validate(r) for r in a.reports],
        user_annotations=[AnnotationOut.model_validate(ann) for ann in a.user_annotations],
    )


class DetectClassRequest(BaseModel):
    class_name: str


@router.post("/{analysis_id}/detect-class")
def detect_class(analysis_id: int, body: DetectClassRequest,
                 request: Request, db: Session = Depends(get_db)):
    """Run Grounding DINO for one specific class (fallback — YOLO covers fixed classes)."""
    a = db.get(Analysis, analysis_id)
    if not a:
        raise HTTPException(404)
    from services.image_service import render_annotated
    from models import UserAnnotation

    # This endpoint only works with Grounding DINO (zero-shot arbitrary classes)
    model_type = getattr(request.app.state, "model_type", "gdino")
    if model_type == "yolo":
        # Can't do arbitrary class detection with YOLO — return empty
        return {"detected": 0, "class_name": body.class_name,
                "note": "Arbitrary class detection not available with YOLO fine-tuned model."}

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

    # Rebuild annotated image
    if a.annotated_path:
        all_dets = db.query(Detection).filter_by(analysis_id=analysis_id).all()
        det_dicts = [
            {"box_x1": d.box_x1, "box_y1": d.box_y1, "box_x2": d.box_x2, "box_y2": d.box_y2,
             "class_name": d.class_name, "score": d.score, "category": d.category, "is_valid": 1}
            for d in all_dets
        ]
        regions = [
            {"box_x1": x.box_x1, "box_y1": x.box_y1, "box_x2": x.box_x2, "box_y2": x.box_y2,
             "class_name": x.class_name}
            for x in db.query(UserAnnotation).filter_by(analysis_id=analysis_id, kind="region").all()
            if x.box_x1 is not None
        ]
        render_annotated(Path(a.image_path), det_dicts, regions, Path(a.annotated_path))

    return {"detected": len(saved), "class_name": body.class_name}
