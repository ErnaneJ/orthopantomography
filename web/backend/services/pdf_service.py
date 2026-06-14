"""
OPG Preliminary Analysis Report — A4 portrait.

This document is an AI-assisted preliminary analysis. It must be reviewed
and validated by a licensed dental professional before clinical use.

Page layout:
  Page 1   Cover: header + exam info + original radiograph
  Page 2   Annotated radiograph + category legend
  Page 3   Findings table
  Page N+  AI pre-report (V1) and Enriched pre-report (V2) if present
"""
import re
from datetime import datetime
from pathlib import Path
from fpdf import FPDF

# ── OPG native aspect ratio (2852 × 1504) ───────────────────────────────────
OPG_AR = 1504 / 2852   # ≈ 0.527  height/width

# ── Colours ─────────────────────────────────────────────────────────────────
NAVY  = (15,  40,  90)
BLUE  = (37,  99,  235)
LBLUE = (239, 246, 255)
DBLUE = (30,  58,  138)
SLT   = (226, 232, 240)   # silver/slate border
DARK  = (15,  23,  42)
MID   = (71,  85,  105)
WHITE = (255, 255, 255)
ERED  = (185, 28,  28)
EGRN  = (21,  128, 61)
EAMB  = (180, 83,  9)

CAT_RGB = {
    "Diseases":     (220, 38,  38),
    "Treatments":   (37,  99,  235),
    "Tooth Status": (22,  163, 74),
    "Anatomy":      (234, 88,  12),
    "Orthodontics": (124, 58,  237),
    "User":         (202, 138, 4),
    "Unknown":      (100, 116, 139),
}

# ── Text helpers ─────────────────────────────────────────────────────────────
_SUBS = [
    ("—", "-"), ("–", "-"),
    ("‘", "'"), ("’", "'"),
    ("“", '"'), ("”", '"'),
    ("•", "*"), ("…", "..."),
    # latin1 escapes for common accented chars
    ("\xe9", "e"), ("\xe8", "e"), ("\xea", "e"), ("\xe0", "a"),
    ("\xe2", "a"), ("\xe7", "c"), ("\xf4", "o"), ("\xfb", "u"),
    ("\xe3", "a"), ("\xf5", "o"), ("\xed", "i"), ("\xfa", "u"),
    ("\xf1", "n"), ("\xe1", "a"), ("\xf3", "o"), ("\xf2", "o"),
]

def _s(txt: str) -> str:
    if not txt:
        return ""
    for src, dst in _SUBS:
        txt = txt.replace(src, dst)
    return txt.encode("latin-1", errors="replace").decode("latin-1")

def _md(txt: str) -> str:
    txt = re.sub(r"^#{1,4}\s*", "", txt, flags=re.MULTILINE)
    txt = re.sub(r"\*\*(.*?)\*\*", r"\1", txt)
    txt = re.sub(r"\*(.*?)\*",    r"\1", txt)
    return txt.strip()

def _clip(txt: str, n: int) -> str:
    return txt[:n] if len(txt) > n else txt

def _wrap(line: str, maxlen: int = 90) -> str:
    out = []
    for w in line.split():
        while len(w) > maxlen:
            out.append(w[:maxlen]); w = w[maxlen:]
        out.append(w)
    return " ".join(out)


# ── PDF class ────────────────────────────────────────────────────────────────
class OPGPdf(FPDF):
    _ew = 174   # effective width (210 - 2*18 mm)

    def __init__(self, meta: dict):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.meta = meta
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(18, 18, 18)

    # ── Running header / footer ──────────────────────────────────────────────
    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 9, "F")
        self.set_xy(18, 2)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*WHITE)
        lbl = _s(f"OPG Preliminary Analysis  |  {self.meta['filename']}  |  Ref #{self.meta['id']}")
        self.cell(self._ew, 5, lbl, align="L")
        self.set_text_color(*DARK)
        self.set_y(13)

    def footer(self):
        self.set_y(-13)
        self.set_draw_color(*SLT)
        self.line(18, self.get_y(), 192, self.get_y())
        self.ln(1)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*MID)
        self.cell(0, 5,
            _s(f"AI-assisted preliminary analysis — not a clinical report — {self.meta['generated']}  |  Page {self.page_no()}"),
            align="C")

    # ── Layout primitives ────────────────────────────────────────────────────
    def rule(self, color=SLT):
        self.set_draw_color(*color)
        self.set_line_width(0.25)
        self.line(18, self.get_y(), 192, self.get_y())
        self.set_line_width(0.2)
        self.ln(3)

    def section(self, title: str, fg=DBLUE):
        self.ln(2)
        # Accent bar + title on same baseline
        bar_y = self.get_y()
        self.set_fill_color(*BLUE)
        self.rect(18, bar_y, 2.5, 7.5, "F")
        self.set_fill_color(*LBLUE)
        self.rect(20.5, bar_y, self._ew - 2.5, 7.5, "F")
        self.set_xy(23, bar_y + 0.5)
        self.set_font("Helvetica", "B", 9.5)
        self.set_text_color(*fg)
        self.cell(self._ew - 5, 6.5, _s(title), align="L")
        self.set_text_color(*DARK)
        self.ln(10)

    def kv(self, key: str, val: str, key_w: float = 36):
        self.set_x(18)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*MID)
        self.cell(key_w, 6, _s(key + ":"))
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*DARK)
        self.cell(self._ew - key_w, 6, _s(_clip(str(val), 80)),
                  new_x="LMARGIN", new_y="NEXT")

    def body(self, text: str):
        """Render a markdown pre-report with headings, bullets and tooth entries."""
        HEADS = {"PANORAMIC", "OVERALL", "TOOTH-BY", "DETECTED",
                 "RECOMMENDED", "RADIOGRAPHIC", "FINDINGS", "SUMMARY",
                 "IMPRESSION", "CLINICAL", "RECOMMENDATIONS"}
        lines = _s(_md(text)).splitlines()
        for raw in lines:
            line = raw.rstrip()
            if not line:
                self.ln(2.5)
                continue
            first = (line.split()[0].rstrip(":,;.")).upper() if line.split() else ""
            if any(line.upper().startswith(h) for h in HEADS) or first in HEADS:
                self.ln(3)
                self.set_x(18)
                self.set_font("Helvetica", "B", 9)
                self.set_text_color(*DBLUE)
                try:
                    self.multi_cell(self._ew, 5.5, _wrap(line))
                except Exception:
                    pass
                self.rule(SLT)
                self.set_text_color(*DARK)
                continue
            m = re.match(r"(Tooth \d+):\s*(.*)", line)
            if m:
                self.ln(0.5)
                self.set_x(18)
                self.set_font("Helvetica", "B", 8.5)
                self.set_text_color(*DARK)
                try:
                    self.cell(28, 5.5, _s(m.group(1)) + ":", ln=False)
                except Exception:
                    pass
                self.set_font("Helvetica", "", 8.5)
                try:
                    self.multi_cell(self._ew - 28, 5.5, _wrap(m.group(2)))
                except Exception:
                    pass
                continue
            if line.startswith("- "):
                self.set_x(22)
                self.set_font("Helvetica", "", 8.5)
                try:
                    self.multi_cell(self._ew - 4, 5.5, "•  " + _wrap(line[2:]))
                except Exception:
                    pass
                continue
            self.set_x(18)
            self.set_font("Helvetica", "", 8.5)
            try:
                self.multi_cell(self._ew, 5.5, _wrap(line))
            except Exception:
                pass
        self.ln(2)


# ── Public API ───────────────────────────────────────────────────────────────
def generate(analysis_id: int, filename: str, image_path: Path,
             annotated_path: Path | None, detections: list[dict],
             reports: list[dict], annotations: list[dict]) -> bytes:

    now = datetime.now()
    meta = {
        "id":        analysis_id,
        "filename":  filename,
        "generated": now.strftime("%Y-%m-%d %H:%M"),
        "date":      now.strftime("%Y-%m-%d"),
    }
    pdf = OPGPdf(meta)

    valid  = [d for d in detections if d.get("is_valid", 1)]
    n_det  = len(valid)
    n_cls  = len({d["class_name"] for d in valid})
    ai_n   = sum(1 for d in valid if d.get("source", "auto") == "auto")
    usr_n  = n_det - ai_n
    avg_c  = (sum(d["score"] for d in valid) / n_det) if n_det else 0.0

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ════════════════════════════════════════════════════════════════════════
    pdf.add_page()

    # Navy header band
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 38, "F")
    pdf.set_fill_color(*BLUE)
    pdf.rect(0, 35, 210, 3, "F")

    pdf.set_xy(18, 7)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 9, "OPG PRELIMINARY ANALYSIS REPORT", align="L")

    pdf.set_xy(18, 17)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(180, 210, 255)
    pdf.cell(0, 6, "Panoramic Dental Radiograph - AI-Assisted Pre-Report", align="L")

    pdf.set_xy(18, 26)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_text_color(140, 175, 220)
    pdf.cell(0, 5,
        "This document is generated by an AI system and must be reviewed by a licensed dentist before clinical use.",
        align="L")

    # Ref number top-right
    pdf.set_xy(0, 7)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(180, 210, 255)
    pdf.cell(192, 5, f"Ref #{analysis_id}", align="R")
    pdf.set_xy(0, 13)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.cell(192, 5, _s(meta["generated"]), align="R")

    pdf.set_text_color(*DARK)
    pdf.set_y(44)

    # ── Exam information box ─────────────────────────────────────────────────
    pdf.section("Exam Information")

    pairs = [
        ("File",          filename),
        ("Exam type",     "Panoramic Radiograph (OPG)"),
        ("Analysis date", meta["date"]),
        ("Total findings", f"{n_det}  (AI: {ai_n}  |  Operator: {usr_n})"),
        ("Unique classes", str(n_cls)),
        ("Avg confidence", f"{avg_c:.1%}"),
    ]
    for k, v in pairs:
        pdf.kv(k, v)
    pdf.ln(3)

    # ── Original radiograph ──────────────────────────────────────────────────
    pdf.section("Original Panoramic Radiograph")
    if image_path.exists():
        img_h = pdf._ew * OPG_AR
        pdf.image(str(image_path), x=18, w=pdf._ew, h=img_h)
        pdf.ln(4)

    # Category summary chips
    cats: dict[str, int] = {}
    for d in valid:
        cats[d["category"]] = cats.get(d["category"], 0) + 1
    if cats:
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*MID)
        pdf.set_x(18)
        pdf.cell(0, 5, "FINDINGS BY CATEGORY:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(18)
        for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
            r, g, b = CAT_RGB.get(cat, (100, 116, 139))
            pdf.set_fill_color(r, g, b)
            pdf.set_text_color(*WHITE)
            pdf.set_font("Helvetica", "B", 6.5)
            pdf.cell(48, 6, _s(f"  {cat}: {cnt}"), fill=True)
            pdf.cell(3, 6, "")
        pdf.ln(10)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 2 — ANNOTATED IMAGE
    # ════════════════════════════════════════════════════════════════════════
    if annotated_path and annotated_path.exists():
        pdf.add_page()
        pdf.section("Annotated Radiograph — Detected Findings")
        img_h = pdf._ew * OPG_AR
        pdf.image(str(annotated_path), x=18, w=pdf._ew, h=img_h)
        pdf.ln(5)

        # Legend
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(*MID)
        pdf.set_x(18)
        pdf.cell(0, 5, "CATEGORY LEGEND:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_x(18)
        for cat, (r, g, b) in CAT_RGB.items():
            pdf.set_fill_color(r, g, b)
            cy = pdf.get_y() + 1.5
            pdf.rect(pdf.get_x(), cy, 5, 4.5, "F")
            pdf.set_x(pdf.get_x() + 7)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(*DARK)
            pdf.cell(40, 7.5, _s(cat))
        pdf.ln(10)

        # Methodology note
        pdf.set_fill_color(248, 250, 252)
        pdf.set_draw_color(*SLT)
        note_y = pdf.get_y()
        pdf.set_x(18)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*MID)
        pdf.multi_cell(pdf._ew, 4.5,
            "Solid coloured boxes = AI detections (Grounding DINO, zero-shot). "
            "Dashed yellow boxes = operator-drawn regions. "
            "Confidence score shown beside each label. "
            "All findings require clinician review before clinical use.",
            align="L")
        pdf.ln(3)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 3 — FINDINGS TABLE
    # ════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section("Detected Findings Table")

    # Column definitions: (header, width mm, align)
    cols = [
        ("#",         7,  "C"),
        ("Finding",  48,  "L"),
        ("Category", 28,  "L"),
        ("Conf.",    16,  "C"),
        ("Source",   16,  "C"),
        ("Box (px)",  30,  "C"),
        ("Status",   15,  "C"),
        ("Valid",    14,  "C"),
    ]
    # Trim to fit: sum = 7+48+28+16+16+30+15+14 = 174 ✓

    # Header row
    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_draw_color(*NAVY)
    pdf.set_x(18)
    for label, w, align in cols:
        pdf.cell(w, 7.5, _s(label), border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_draw_color(*SLT)
    pdf.set_line_width(0.15)

    for i, det in enumerate(detections, 1):
        is_valid = bool(det.get("is_valid", 1))
        score = det["score"]
        cat = det["category"]
        cr, cg, cb = CAT_RGB.get(cat, (100, 116, 139))

        fill_bg = (248, 250, 252) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*fill_bg)

        # Confidence colour
        if score >= 0.80:
            scr, scg, scb = EGRN
        elif score >= 0.55:
            scr, scg, scb = EAMB
        else:
            scr, scg, scb = ERED

        box = _s(f"({det['box_x1']:.0f},{det['box_y1']:.0f})-({det['box_x2']:.0f},{det['box_y2']:.0f})")

        pdf.set_x(18)
        pdf.set_text_color(*MID)
        pdf.cell(cols[0][1], 6, str(i), border=1, align="C", fill=True)

        pdf.set_text_color(*DARK)
        pdf.cell(cols[1][1], 6, _s(_clip(det["class_name"], 25)), border=1, fill=True)

        # Category cell: left 3mm accent, rest text
        pdf.set_fill_color(cr, cg, cb)
        pdf.cell(3, 6, "", border="LTB", fill=True)
        pdf.set_fill_color(*fill_bg)
        pdf.set_text_color(*DARK)
        pdf.cell(cols[2][1] - 3, 6, _s(_clip(cat, 16)), border="RTB", fill=True)

        pdf.set_text_color(scr, scg, scb)
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(cols[3][1], 6, f"{score:.0%}", border=1, align="C", fill=True)
        pdf.set_font("Helvetica", "", 7)

        pdf.set_text_color(*DARK)
        src = "AI" if det.get("source", "auto") == "auto" else "Operator"
        pdf.cell(cols[4][1], 6, src, border=1, align="C", fill=True)
        pdf.cell(cols[5][1], 6, _clip(box, 22), border=1, align="C", fill=True)

        # Status / validity
        if is_valid:
            pdf.set_text_color(*EGRN)
            pdf.cell(cols[6][1], 6, "Valid",    border=1, align="C", fill=True)
            pdf.set_text_color(*EGRN)
            pdf.cell(cols[7][1], 6, "Yes",      border=1, align="C", fill=True)
        else:
            pdf.set_text_color(*ERED)
            pdf.cell(cols[6][1], 6, "Removed",  border=1, align="C", fill=True)
            pdf.set_text_color(*ERED)
            pdf.cell(cols[7][1], 6, "No",       border=1, align="C", fill=True)

        pdf.ln()

    pdf.set_text_color(*DARK)
    pdf.ln(3)

    # Summary stats row
    pdf.set_fill_color(*LBLUE)
    pdf.set_x(18)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(*MID)
    pdf.multi_cell(pdf._ew, 5.5,
        _s(f"Total valid findings: {n_det}   |   AI: {ai_n}   Operator: {usr_n}"
           f"   |   Unique classes: {n_cls}   |   Avg confidence: {avg_c:.1%}"),
        fill=True, align="L")
    pdf.ln(4)

    # Operator annotations
    text_anns = [a for a in annotations if a.get("content")]
    if text_anns:
        pdf.section("Operator Annotations")
        for ann in text_anns:
            kind = {"text": "Note", "transcription": "Voice", "region": "Region"}.get(ann["kind"], ann["kind"])
            prefix = f"[{kind}]"
            if ann.get("class_name"):
                prefix += f"  {ann['class_name']}"
            pdf.set_x(18)
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_text_color(*DBLUE)
            pdf.cell(0, 5.5, _s(prefix), new_x="LMARGIN", new_y="NEXT")
            pdf.set_x(22)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*DARK)
            try:
                pdf.multi_cell(pdf._ew - 4, 5, _s(_wrap(ann.get("content") or "")))
            except Exception:
                pass
            pdf.ln(1.5)

    # ════════════════════════════════════════════════════════════════════════
    # PAGE(S) — REPORTS
    # ════════════════════════════════════════════════════════════════════════
    report_labels = {
        1: "AI Pre-Report - Version 1 (Automatic)",
        2: "AI Pre-Report - Version 2 (Operator-Enriched)",
    }
    for rep in sorted(reports, key=lambda r: r["version"]):
        pdf.add_page()
        pdf.section(report_labels.get(rep["version"], f"Pre-Report V{rep['version']}"))
        pdf.body(rep["content"])

        # Disclaimer + signature area
        pdf.ln(6)
        pdf.rule(SLT)
        pdf.set_fill_color(*LBLUE)
        pdf.set_x(18)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*MID)
        pdf.multi_cell(pdf._ew, 4.5,
            "DISCLAIMER: This pre-report was generated automatically by an AI system using "
            "zero-shot object detection and a large language model. It is intended as a "
            "decision-support tool only. The responsible dental professional must review, "
            "validate, and sign the final clinical report. AI findings may contain false "
            "positives or false negatives.",
            fill=True, align="L")
        pdf.ln(8)

        # Signature lines
        pdf.set_x(18)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*MID)
        pdf.cell(87, 5, "Reviewing Dentist / CRO", align="C")
        pdf.cell(87, 5, "Date of Review", align="C")
        pdf.ln(14)
        pdf.set_draw_color(*SLT)
        pdf.line(22,  pdf.get_y(), 100, pdf.get_y())
        pdf.line(112, pdf.get_y(), 190, pdf.get_y())

    return bytes(pdf.output())
