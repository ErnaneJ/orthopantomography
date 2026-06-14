import { useEffect, useState, useCallback, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { AnalysisDetail } from "../types";
import FindingsList from "../components/FindingsList";
import AnnotationCanvas, { CLASS_NAMES } from "../components/AnnotationCanvas";
import EnrichPanel from "../components/EnrichPanel";
import ReportTabs from "../components/ReportTabs";
import ImageLightbox from "../components/ImageLightbox";
import { formatDate } from "../utils/date";

type PendingRegion = { x1: number; y1: number; x2: number; y2: number; cls: string };

function getPhase(data: AnalysisDetail) {
  if (data.status === "error") return "error";
  if (data.status === "done" || data.status === "enriching") return "done";
  if (data.detections.length > 0) return "reporting";
  return "detecting";
}

const PHASE_LABEL: Record<string, string> = {
  detecting: "Detecting pathologies...",
  reporting: "Generating pre-report...",
  done: "",
  error: "Error",
};

export default function DetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const analysisId = Number(id);

  const [data, setData]               = useState<AnalysisDetail | null>(null);
  const [showAnnotated, setShowAnnotated] = useState(false);
  const [selectedClass, setSelectedClass] = useState(CLASS_NAMES[0]);
  const [pendingRegions, setPendingRegions] = useState<PendingRegion[]>([]);
  const [enriching, setEnriching]     = useState(false);
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);

  const enrichIvRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = useCallback(async () => {
    const d = await api.getAnalysis(analysisId);
    setData(d);
    return d;
  }, [analysisId]);

  useEffect(() => {
    let alive = true;
    let iv: ReturnType<typeof setInterval> | null = null;

    const poll = async () => {
      if (!alive || document.hidden) return;
      try {
        const d = await api.getAnalysis(analysisId);
        if (!alive) return;
        setData(d);
        if (d.status === "done" || d.status === "error") {
          if (iv) clearInterval(iv);
        }
      } catch (e: any) {
        if (!alive) return;
        if (e?.message === "404") { if (iv) clearInterval(iv); navigate("/"); }
      }
    };

    poll();
    iv = setInterval(poll, 2000);

    const onVisibility = () => { if (!document.hidden) poll(); };
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      alive = false;
      if (iv) clearInterval(iv);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [analysisId, navigate]);

  useEffect(() => {
    return () => { if (enrichIvRef.current) clearInterval(enrichIvRef.current); };
  }, []);

  const handleRegionDrawn = async (box: { x1: number; y1: number; x2: number; y2: number }, cls: string) => {
    setPendingRegions(p => [...p, { ...box, cls }]);
    await api.addAnnotation(analysisId, { kind: "region", class_name: cls, ...box });
    load();
  };

  const handleEnrich = async (_classNames: string[], contextNotes: string) => {
    setEnriching(true);
    if (contextNotes.trim())
      await api.addAnnotation(analysisId, { kind: "text", content: contextNotes });
    await api.enrich(analysisId, [], contextNotes);
    setPendingRegions([]);

    if (enrichIvRef.current) clearInterval(enrichIvRef.current);
    enrichIvRef.current = setInterval(async () => {
      try {
        const d = await load();
        if (d.status === "done") {
          if (enrichIvRef.current) { clearInterval(enrichIvRef.current); enrichIvRef.current = null; }
          setEnriching(false);
        }
      } catch { /* ignore */ }
    }, 2000);
  };

  if (!data)
    return (
      <div className="flex flex-col items-center justify-center py-32 gap-3">
        <div className="w-10 h-10 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
        <p className="text-slate-400 text-sm">Loading analysis...</p>
      </div>
    );

  const phase = getPhase(data);
  const processing = phase === "detecting" || phase === "reporting" || enriching;
  const validDets  = data.detections.filter(d => d.is_valid);
  const hasV2      = data.reports.some(r => r.version === 2);

  const currentImageUrl = showAnnotated && data.annotated_url ? data.annotated_url : data.image_url;
  const showDetectionOverlays = !showAnnotated;
  const canDraw = showAnnotated;

  return (
    <>
      {lightboxSrc && (
        <ImageLightbox
          src={lightboxSrc}
          alt={data.filename}
          onClose={() => setLightboxSrc(null)}
          showFDI
        />
      )}

      <div className="space-y-5">
        {/* Header */}
        <div className="flex items-start justify-between gap-4">
          <div>
            <button onClick={() => navigate("/")}
              className="text-xs text-slate-400 hover:text-blue-600 flex items-center gap-1 mb-1 transition-colors">
              ← All analyses
            </button>
            <h2 className="text-xl font-bold text-slate-800">{data.filename}</h2>
            <p className="text-xs text-slate-400 mt-0.5">#{data.id} · {formatDate(data.created_at)}</p>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            {processing && (
              <span className="flex items-center gap-2 text-amber-600 text-sm font-medium bg-amber-50 px-3 py-1.5 rounded-full border border-amber-200">
                <span className="w-3 h-3 border-2 border-amber-200 border-t-amber-500 rounded-full animate-spin shrink-0" />
                {enriching ? "Enriching pre-report..." : PHASE_LABEL[phase]}
              </span>
            )}
            {data.status === "done" && (
              <a href={api.pdfUrl(analysisId)} target="_blank" rel="noopener noreferrer"
                className="btn-primary flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414A1 1 0 0119 9.414V19a2 2 0 01-2 2z" />
                </svg>
                Download PDF
              </a>
            )}
          </div>
        </div>

        {/* Progress steps */}
        <div className="card py-3.5 px-5">
          <div className="flex items-center gap-2 flex-wrap">
            {[
              { label: "Detection",   icon: "🔍", done: data.detections.length > 0 },
              { label: "Pre-report",  icon: "🤖", done: data.reports.some(r => r.version === 1) },
              { label: "Enriched",    icon: "✏️", done: hasV2 },
            ].map(({ label, icon, done }, i) => (
              <div key={i} className="flex items-center gap-2">
                {i > 0 && <div className={`w-6 h-px ${done ? "bg-blue-300" : "bg-slate-200"}`} />}
                <span className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold transition-colors
                  ${done ? "bg-blue-50 text-blue-700" : "bg-slate-100 text-slate-400"}`}>
                  {icon} {label} {done && "✓"}
                </span>
              </div>
            ))}
            {phase === "detecting" && (
              <span className="ml-auto flex items-center gap-1.5 text-xs text-amber-600">
                <span className="w-2.5 h-2.5 border-2 border-amber-200 border-t-amber-500 rounded-full animate-spin" />
                Running Stage 1...
              </span>
            )}
            {phase === "reporting" && (
              <span className="ml-auto flex items-center gap-1.5 text-xs text-amber-600">
                <span className="w-2.5 h-2.5 border-2 border-amber-200 border-t-amber-500 rounded-full animate-spin" />
                Running Stage 3...
              </span>
            )}
          </div>
        </div>

        {/* Error */}
        {data.status === "error" && (
          <div className="card border-red-200 bg-red-50 flex items-start gap-3">
            <span className="text-red-400 text-lg shrink-0">⚠️</span>
            <div>
              <p className="font-semibold text-red-700 text-sm">Processing error</p>
              <p className="text-red-600 text-xs mt-0.5 font-mono">{data.error_msg}</p>
            </div>
          </div>
        )}

        {/* Main grid */}
        <div className="grid grid-cols-1 xl:grid-cols-5 gap-5">
          {/* Left: canvas 3/5 */}
          <div className="xl:col-span-3 space-y-4">

            {/* View toggle + class selector */}
            <div className="card py-3 px-4 flex items-center justify-between gap-3 flex-wrap">
              <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-xl shrink-0">
                {[
                  { label: "Original",   val: false },
                  { label: "Annotated",  val: true, disabled: !data.annotated_url },
                ].map(opt => (
                  <button key={opt.label} disabled={opt.disabled}
                    onClick={() => setShowAnnotated(opt.val)}
                    className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all whitespace-nowrap
                      ${showAnnotated === opt.val ? "bg-white text-slate-800 shadow-sm" : "text-slate-500 hover:text-slate-700"}
                      ${opt.disabled ? "opacity-40 cursor-not-allowed" : ""}`}>
                    {opt.label}
                  </button>
                ))}
              </div>

              {canDraw ? (
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <label className="text-xs text-slate-500 font-medium shrink-0">Class:</label>
                  <select value={selectedClass} onChange={e => setSelectedClass(e.target.value)}
                    className="select-styled !py-1.5 !text-xs border-slate-200 flex-1 min-w-0">
                    {CLASS_NAMES.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
              ) : (
                <span className="text-xs text-slate-400 italic">
                  Switch to <strong>Annotated</strong> to draw regions
                </span>
              )}
            </div>

            {/* Canvas */}
            <div className="card p-0 overflow-hidden">
              <AnnotationCanvas
                imageUrl={currentImageUrl}
                detections={data.detections}
                userAnnotations={data.user_annotations}
                selectedClass={selectedClass}
                showOverlays={showDetectionOverlays}
                readOnly={!canDraw}
                onRegionDrawn={handleRegionDrawn}
              />
              <div className="flex items-center justify-between px-4 py-2 bg-slate-50 border-t border-slate-100">
                <p className="text-xs text-slate-400">
                  {canDraw
                    ? "Drag on image to mark a region · Select class before drawing"
                    : "Read-only — switch to \"Annotated\" to draw regions"}
                </p>
                <button onClick={() => setLightboxSrc(currentImageUrl)}
                  className="flex items-center gap-1.5 text-xs text-blue-600 hover:text-blue-800 font-medium transition-colors">
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
                  </svg>
                  Fullscreen
                </button>
              </div>
            </div>

            {/* Enrich */}
            {(data.status === "done" || hasV2) && (
              <EnrichPanel
                analysisId={analysisId}
                pendingRegions={pendingRegions}
                onEnrich={handleEnrich}
                onUpdate={load}
                enriching={enriching}
                onClearRegions={() => setPendingRegions([])}
              />
            )}
          </div>

          {/* Right: sidebar 2/5 */}
          <div className="xl:col-span-2 space-y-4">

            {data.detections.length > 0 && (
              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: "Findings",  value: validDets.length, color: "text-blue-600" },
                  { label: "Classes",   value: new Set(validDets.map(d => d.class_name)).size, color: "text-purple-600" },
                  { label: "Versions",  value: data.reports.length, color: "text-emerald-600" },
                ].map(s => (
                  <div key={s.label} className="card text-center py-3">
                    <p className={`text-2xl font-bold ${s.color}`}>{s.value}</p>
                    <p className="text-xs text-slate-400 mt-0.5">{s.label}</p>
                  </div>
                ))}
              </div>
            )}

            {data.detections.length > 0 && (
              <div className="card">
                <p className="section-title">Detected findings</p>
                {phase === "detecting" && (
                  <div className="flex items-center gap-2 text-slate-400 text-xs py-1 mb-2">
                    <div className="w-3 h-3 border-2 border-slate-200 border-t-blue-400 rounded-full animate-spin" />
                    Detecting...
                  </div>
                )}
                <FindingsList detections={data.detections} onUpdate={load} />
              </div>
            )}

            {phase === "detecting" && data.detections.length === 0 && (
              <div className="card flex items-center justify-center gap-3 py-8 text-slate-400">
                <div className="w-5 h-5 border-2 border-slate-200 border-t-blue-400 rounded-full animate-spin shrink-0" />
                <span className="text-sm">Detecting findings...</span>
              </div>
            )}

            {data.user_annotations.filter(a => a.content).length > 0 && (
              <div className="card">
                <p className="section-title">Operator annotations</p>
                <div className="space-y-2">
                  {data.user_annotations.filter(a => a.content).map(ann => (
                    <div key={ann.id} className="flex items-start justify-between gap-2 bg-amber-50/60 rounded-xl p-3 border border-amber-100">
                      <div className="flex items-start gap-2 min-w-0">
                        <span className="shrink-0 text-sm mt-0.5">
                          {ann.kind === "transcription" ? "🎤" : ann.kind === "region" ? "📍" : "📝"}
                        </span>
                        <p className="text-xs text-slate-700 leading-relaxed">{ann.content}</p>
                      </div>
                      <button onClick={async () => { await api.deleteAnnotation(analysisId, ann.id); load(); }}
                        className="text-slate-300 hover:text-red-400 shrink-0 transition-colors text-sm">✕</button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {data.reports.length > 0 && (
              <div className="card">
                <p className="section-title">Pre-report</p>
                <ReportTabs reports={data.reports} />
              </div>
            )}

            {phase === "reporting" && data.reports.length === 0 && (
              <div className="card flex items-center gap-3 py-6 text-slate-400">
                <div className="w-5 h-5 border-2 border-slate-200 border-t-blue-400 rounded-full animate-spin shrink-0" />
                <span className="text-sm">Generating pre-report...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
