import { useEffect, useRef, useState, useCallback } from "react";

interface Props {
  src: string;
  alt?: string;
  onClose: () => void;
  showFDI?: boolean;
}

const FDI_UPPER = ["18","17","16","15","14","13","12","11","21","22","23","24","25","26","27","28"];
const FDI_LOWER = ["48","47","46","45","44","43","42","41","31","32","33","34","35","36","37","38"];

export default function ImageLightbox({ src, alt = "", onClose, showFDI = false }: Props) {
  const [scale, setScale]       = useState(1);
  const [pos, setPos]           = useState({ x: 0, y: 0 });
  const [showGuide, setShowGuide] = useState(showFDI);
  const panRef = useRef({ dragging: false, lastX: 0, lastY: 0 });

  useEffect(() => {
    const fn = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, [onClose]);

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setScale(s => Math.min(8, Math.max(0.25, s * (e.deltaY < 0 ? 1.12 : 0.89))));
  }, []);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    panRef.current = { dragging: true, lastX: e.clientX, lastY: e.clientY };
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!panRef.current.dragging) return;
    const dx = e.clientX - panRef.current.lastX;
    const dy = e.clientY - panRef.current.lastY;
    panRef.current.lastX = e.clientX;
    panRef.current.lastY = e.clientY;
    setPos(p => ({ x: p.x + dx, y: p.y + dy }));
  };

  const onPointerUp = () => { panRef.current.dragging = false; };

  const reset  = () => { setScale(1); setPos({ x: 0, y: 0 }); };
  const zoomIn  = () => setScale(s => Math.min(8, s * 1.3));
  const zoomOut = () => setScale(s => Math.max(0.25, s / 1.3));

  return (
    <div className="fixed inset-0 z-[100] flex flex-col select-none"
      style={{ background: "rgba(0,0,0,0.82)", backdropFilter: "blur(6px)" }}
      onClick={onClose}>

      {/* Toolbar */}
      <div className="flex items-center gap-3 px-5 py-3 bg-black/50 backdrop-blur-sm shrink-0"
        onClick={e => e.stopPropagation()}>
        <span className="text-white/60 text-sm font-medium truncate max-w-xs">{alt}</span>
        <div className="ml-auto flex items-center gap-2">
          <button onClick={zoomOut} className="w-8 h-8 rounded-lg bg-white/10 hover:bg-white/20 text-white flex items-center justify-center text-lg transition">−</button>
          <span className="text-white/70 text-sm w-14 text-center font-mono">{(scale * 100).toFixed(0)}%</span>
          <button onClick={zoomIn}  className="w-8 h-8 rounded-lg bg-white/10 hover:bg-white/20 text-white flex items-center justify-center text-lg transition">+</button>
          <button onClick={reset}   className="px-3 h-8 rounded-lg bg-white/10 hover:bg-white/20 text-white/70 text-xs font-medium transition">Reset</button>
          <div className="w-px h-5 bg-white/20 mx-1" />
          <button onClick={() => setShowGuide(g => !g)}
            className={`px-3 h-8 rounded-lg text-xs font-medium transition
              ${showGuide ? "bg-blue-500/70 text-white" : "bg-white/10 hover:bg-white/20 text-white/70"}`}>
            FDI Guide
          </button>
          <div className="w-px h-5 bg-white/20 mx-1" />
          <button onClick={onClose}
            className="w-8 h-8 rounded-lg bg-white/10 hover:bg-red-500/60 text-white flex items-center justify-center text-lg transition">✕</button>
        </div>
      </div>

      {/* Image area — pointer events for reliable drag */}
      <div className="flex-1 overflow-hidden flex items-center justify-center relative"
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
        onClick={e => e.stopPropagation()}
        style={{ cursor: panRef.current.dragging ? "grabbing" : "grab", background: "rgba(15,23,42,0.6)" }}>
        <img src={src} alt={alt} draggable={false}
          style={{
            transform: `translate(${pos.x}px, ${pos.y}px) scale(${scale})`,
            transition: panRef.current.dragging ? "none" : "transform 0.08s ease",
            maxWidth: "none",
            pointerEvents: "none",
          }}
          className="max-h-[82vh] object-contain select-none shadow-2xl rounded-sm" />
      </div>

      {/* FDI guide */}
      {showGuide && (
        <div className="shrink-0 bg-black/70 backdrop-blur-sm px-5 py-3" onClick={e => e.stopPropagation()}>
          <div className="max-w-2xl mx-auto">
            <p className="text-white/50 text-[10px] font-bold uppercase tracking-widest mb-2 text-center">FDI Tooth Notation</p>
            <div className="flex gap-px mb-1 justify-center">
              {FDI_UPPER.map((t, i) => (
                <div key={t}
                  className={`w-8 h-7 flex items-center justify-center text-[10px] font-bold rounded
                    ${i < 8 ? "bg-blue-900/60 text-blue-300" : "bg-purple-900/60 text-purple-300"}`}>
                  {t}
                </div>
              ))}
            </div>
            <div className="flex gap-px justify-center">
              {FDI_LOWER.map((t, i) => (
                <div key={t}
                  className={`w-8 h-7 flex items-center justify-center text-[10px] font-bold rounded
                    ${i < 8 ? "bg-orange-900/60 text-orange-300" : "bg-green-900/60 text-green-300"}`}>
                  {t}
                </div>
              ))}
            </div>
            <div className="flex justify-center gap-6 mt-2">
              {[["bg-blue-700","Q1 (upper right)"],["bg-purple-700","Q2 (upper left)"],
                ["bg-orange-700","Q3 (lower left)"],["bg-green-700","Q4 (lower right)"]].map(([color, label]) => (
                <span key={label} className="flex items-center gap-1 text-[9px] text-white/50">
                  <span className={`w-2 h-2 rounded-sm ${color}`} />{label}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="text-center py-1.5 text-white/30 text-[10px] shrink-0">
        Scroll to zoom · Drag to pan · ESC to close
      </div>
    </div>
  );
}
