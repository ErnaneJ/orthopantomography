from datetime import datetime, timezone
from pydantic import BaseModel, field_serializer


def _utc_iso(dt: datetime) -> str:
    """Return ISO 8601 with explicit Z so JS parses it as UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class DetectionOut(BaseModel):
    id: int
    class_name: str
    score: float
    box_x1: float; box_y1: float; box_x2: float; box_y2: float
    category: str
    source: str
    is_valid: int
    model_config = {"from_attributes": True}


class ReportOut(BaseModel):
    id: int
    version: int
    content: str
    model: str
    created_at: datetime
    model_config = {"from_attributes": True}

    @field_serializer("created_at")
    def ser_created_at(self, v: datetime) -> str: return _utc_iso(v)


class AnnotationOut(BaseModel):
    id: int
    kind: str
    content: str | None
    class_name: str | None
    box_x1: float | None; box_y1: float | None
    box_x2: float | None; box_y2: float | None
    created_at: datetime
    model_config = {"from_attributes": True}

    @field_serializer("created_at")
    def ser_created_at(self, v: datetime) -> str: return _utc_iso(v)


class AnalysisListItem(BaseModel):
    id: int
    filename: str
    status: str
    created_at: datetime
    image_url: str
    model_config = {"from_attributes": True}

    @field_serializer("created_at")
    def ser_created_at(self, v: datetime) -> str: return _utc_iso(v)


class AnalysisDetail(BaseModel):
    id: int
    filename: str
    status: str
    error_msg: str | None
    created_at: datetime
    image_url: str
    annotated_url: str | None
    detections: list[DetectionOut]
    reports: list[ReportOut]
    user_annotations: list[AnnotationOut]
    model_config = {"from_attributes": True}

    @field_serializer("created_at")
    def ser_created_at(self, v: datetime) -> str: return _utc_iso(v)


class AnnotationCreate(BaseModel):
    kind: str
    content: str | None = None
    class_name: str | None = None
    box_x1: float | None = None
    box_y1: float | None = None
    box_x2: float | None = None
    box_y2: float | None = None


class EnrichRequest(BaseModel):
    class_names: list[str]
    context_notes: str
