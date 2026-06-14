from datetime import datetime
from sqlalchemy import Integer, String, Float, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base


class Analysis(Base):
    __tablename__ = "analyses"
    id:             Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename:       Mapped[str]      = mapped_column(String, nullable=False)
    image_path:     Mapped[str]      = mapped_column(String, nullable=False)
    annotated_path: Mapped[str|None] = mapped_column(String, nullable=True)
    status:         Mapped[str]      = mapped_column(String, default="pending")
    error_msg:      Mapped[str|None] = mapped_column(Text, nullable=True)
    created_at:     Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at:     Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    detections:       Mapped[list["Detection"]]      = relationship(back_populates="analysis", cascade="all, delete-orphan")
    reports:          Mapped[list["Report"]]          = relationship(back_populates="analysis", cascade="all, delete-orphan")
    user_annotations: Mapped[list["UserAnnotation"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"
    id:          Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int]   = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"))
    class_name:  Mapped[str]   = mapped_column("class", String, nullable=False)
    raw_label:   Mapped[str|None] = mapped_column(String, nullable=True)
    score:       Mapped[float] = mapped_column(Float, nullable=False)
    box_x1:      Mapped[float] = mapped_column(Float, nullable=False)
    box_y1:      Mapped[float] = mapped_column(Float, nullable=False)
    box_x2:      Mapped[float] = mapped_column(Float, nullable=False)
    box_y2:      Mapped[float] = mapped_column(Float, nullable=False)
    category:    Mapped[str]   = mapped_column(String, nullable=False)
    source:      Mapped[str]   = mapped_column(String, default="auto")
    is_valid:    Mapped[int]   = mapped_column(Integer, default=1)

    analysis: Mapped["Analysis"] = relationship(back_populates="detections")


class Report(Base):
    __tablename__ = "reports"
    id:            Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id:   Mapped[int]      = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"))
    version:       Mapped[int]      = mapped_column(Integer, nullable=False)
    content:       Mapped[str]      = mapped_column(Text, nullable=False)
    model:         Mapped[str]      = mapped_column(String, nullable=False)
    input_tokens:  Mapped[int|None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int|None] = mapped_column(Integer, nullable=True)
    created_at:    Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    analysis: Mapped["Analysis"] = relationship(back_populates="reports")


class UserAnnotation(Base):
    __tablename__ = "user_annotations"
    id:          Mapped[int]      = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int]      = mapped_column(ForeignKey("analyses.id", ondelete="CASCADE"))
    kind:        Mapped[str]      = mapped_column(String, nullable=False)
    content:     Mapped[str|None] = mapped_column(Text, nullable=True)
    class_name:  Mapped[str|None] = mapped_column(String, nullable=True)
    box_x1:      Mapped[float|None] = mapped_column(Float, nullable=True)
    box_y1:      Mapped[float|None] = mapped_column(Float, nullable=True)
    box_x2:      Mapped[float|None] = mapped_column(Float, nullable=True)
    box_y2:      Mapped[float|None] = mapped_column(Float, nullable=True)
    created_at:  Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    analysis: Mapped["Analysis"] = relationship(back_populates="user_annotations")
