import { useRef, useState, useLayoutEffect, useEffect } from "react";
import { Stage, Layer, Image as KonvaImage, Rect, Text, Group } from "react-konva";
import type { Detection, Annotation } from "../types";

const CAT_COLORS: Record<string, string> = {
  Diseases: "#EF4444", Treatments: "#3B82F6", "Tooth Status": "#10B981",
  Anatomy: "#F59E0B", Orthodontics: "#8B5CF6", User: "#EAB308", Unknown: "#94A3B8",
};

export const CLASS_NAMES = [
  "Caries","Crown","Filling","Implant","Malaligned","Mandibular Canal",
  "Missing teeth","Periapical lesion","Retained root","Root Canal Treatment",
  "Root Piece","Impacted tooth","Maxillary sinus","Bone Loss","Fractured teeth",
  "Permanent Teeth","Supra Eruption","TAD","Abutment","Attrition","Bone defect",
  "Gingival former","Metal band","Orthodontic brackets","Permanent retainer",
  "Post-core","Plating","Wire","Cyst","Root resorption","Primary teeth",
];

interface DrawRect { x: number; y: number; w: number; h: number }
interface Props {
  imageUrl: string;
  detections: Detection[];
  userAnnotations: Annotation[];
  selectedClass: string;
  showOverlays: boolean;
  readOnly?: boolean;
  onRegionDrawn: (box: { x1: number; y1: number; x2: number; y2: number }, cls: string) => void;
}

export default function AnnotationCanvas({
  imageUrl, detections, userAnnotations, selectedClass,
  showOverlays, readOnly = false, onRegionDrawn,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dim, setDim] = useState({ w: 760, h: 400 });
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [drawing, setDrawing] = useState<DrawRect | null>(null);
  const [startPt, setStartPt] = useState<{ x: number; y: number } | null>(null);
  // Native image dimensions for accurate coordinate mapping
  const nativeDimRef = useRef({ w: 2852, h: 1504 });

  useLayoutEffect(() => {
    const measure = () => {
      const el = containerRef.current;
      if (!el) return;
      const w = el.getBoundingClientRect().width;
      const { w: nw, h: nh } = nativeDimRef.current;
      if (w > 0) setDim({ w, h: Math.round(w * nh / nw) });
    };
    measure();
    const obs = new ResizeObserver(measure);
    if (containerRef.current) obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const image = new window.Image();
    image.crossOrigin = "anonymous";
    image.src = imageUrl;
    image.onload = () => {
      setImg(image);
      const nw = image.naturalWidth || 2852;
      const nh = image.naturalHeight || 1504;
      nativeDimRef.current = { w: nw, h: nh };
      if (containerRef.current) {
        const containerW = containerRef.current.getBoundingClientRect().width;
        if (containerW > 0) setDim({ w: containerW, h: Math.round(containerW * nh / nw) });
      }
    };
    return () => { image.onload = null; };
  }, [imageUrl]);

  const nw = nativeDimRef.current.w;
  const nh = nativeDimRef.current.h;
  const sx = dim.w / nw, sy = dim.h / nh;
  const td = (v: number, a: "x" | "y") => v * (a === "x" ? sx : sy);
  const to = (v: number, a: "x" | "y") => v / (a === "x" ? sx : sy);
  const gp = (e: any) => e.target.getStage()?.getPointerPosition() ?? { x: 0, y: 0 };

  const onMouseDown = (e: any) => {
    if (readOnly) return;
    const p = gp(e); setStartPt(p); setDrawing({ x: p.x, y: p.y, w: 0, h: 0 });
  };
  const onMouseMove = (e: any) => {
    if (readOnly || !startPt) return;
    const p = gp(e);
    setDrawing({ x: Math.min(startPt.x, p.x), y: Math.min(startPt.y, p.y), w: Math.abs(p.x - startPt.x), h: Math.abs(p.y - startPt.y) });
  };
  const onMouseUp = () => {
    if (readOnly) return;
    if (!drawing || drawing.w < 8 || drawing.h < 8) { setDrawing(null); setStartPt(null); return; }
    onRegionDrawn({ x1: to(drawing.x, "x"), y1: to(drawing.y, "y"), x2: to(drawing.x + drawing.w, "x"), y2: to(drawing.y + drawing.h, "y") }, selectedClass);
    setDrawing(null); setStartPt(null);
  };

  const classCount: Record<string, number> = {};
  const indexedDets = detections.map(d => {
    classCount[d.class_name] = (classCount[d.class_name] ?? 0) + 1;
    return { ...d, idx: classCount[d.class_name] };
  });
  const multiClass = new Set(Object.entries(classCount).filter(([, c]) => c > 1).map(([k]) => k));
  const regions = userAnnotations.filter(a => a.kind === "region" && a.box_x1 != null);

  const STROKE = Math.max(1.5, dim.w / 950);
  const FONT   = Math.max(10, Math.round(dim.w / 95));
  const LABEL_H = FONT + 6;

  return (
    <div ref={containerRef}
      className={`w-full rounded-xl overflow-hidden border border-slate-200 shadow-sm ${readOnly ? "cursor-default" : "cursor-crosshair"}`}>
      <Stage width={dim.w} height={dim.h}
        onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp}>
        <Layer>
          {img && <KonvaImage image={img} width={dim.w} height={dim.h} />}

          {/* AI detection overlays — only in original mode */}
          {showOverlays && indexedDets.map(det => {
            const c = CAT_COLORS[det.category] ?? "#94A3B8";
            const x = td(det.box_x1, "x"), y = td(det.box_y1, "y");
            const w = td(det.box_x2 - det.box_x1, "x"), h = td(det.box_y2 - det.box_y1, "y");
            const label = multiClass.has(det.class_name)
              ? `${det.class_name} #${det.idx} ${Math.round(det.score * 100)}%`
              : `${det.class_name} ${Math.round(det.score * 100)}%`;
            const lw = label.length * (FONT * 0.6) + 8;
            const ly = Math.max(0, y - LABEL_H);
            return (
              <Group key={det.id}>
                <Rect x={x} y={y} width={w} height={h}
                  stroke={c} strokeWidth={STROKE} fill={c} opacity={0.18} cornerRadius={2} />
                <Rect x={x} y={ly} width={lw} height={LABEL_H}
                  fill={c} cornerRadius={[2, 2, 0, 0]} />
                <Text x={x + 4} y={ly + 3} text={label}
                  fontSize={FONT} fill="white" fontStyle="bold" />
              </Group>
            );
          })}

          {/* User regions — always visible regardless of mode */}
          {regions.map(ann => (
            <Group key={ann.id}>
              <Rect
                x={td(ann.box_x1!, "x")} y={td(ann.box_y1!, "y")}
                width={td(ann.box_x2! - ann.box_x1!, "x")} height={td(ann.box_y2! - ann.box_y1!, "y")}
                stroke="#EAB308" strokeWidth={STROKE} dash={[6, 3]} fill="#FEF08A" opacity={0.25} cornerRadius={2} />
              <Text x={td(ann.box_x1!, "x") + 4} y={td(ann.box_y1!, "y") + 4}
                text={`✏ ${ann.class_name || "User"}`}
                fontSize={FONT} fill="#92400E" fontStyle="bold" />
            </Group>
          ))}

          {/* Active drawing rect */}
          {!readOnly && drawing && (
            <Rect x={drawing.x} y={drawing.y} width={drawing.w} height={drawing.h}
              stroke="#EAB308" strokeWidth={STROKE} dash={[5, 3]}
              fill="#FEF08A" opacity={0.3} cornerRadius={2} />
          )}
        </Layer>
      </Stage>
    </div>
  );
}
