import { api } from "../api/client";
import type { Detection } from "../types";

const CAT_COLOR: Record<string, string> = {
  Diseases:      "bg-red-50 text-red-700 ring-1 ring-red-200",
  Treatments:    "bg-blue-50 text-blue-700 ring-1 ring-blue-200",
  "Tooth Status":"bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200",
  Anatomy:       "bg-amber-50 text-amber-700 ring-1 ring-amber-200",
  Orthodontics:  "bg-purple-50 text-purple-700 ring-1 ring-purple-200",
  User:          "bg-yellow-50 text-yellow-700 ring-1 ring-yellow-200",
  Unknown:       "bg-slate-100 text-slate-500 ring-1 ring-slate-200",
};

const CONF_BAR = (s: number) => ({
  bar: s >= 0.7 ? "bg-emerald-400" : s >= 0.45 ? "bg-amber-400" : "bg-slate-300",
  text: s >= 0.7 ? "text-emerald-600" : s >= 0.45 ? "text-amber-600" : "text-slate-400",
});

interface Props { detections: Detection[]; onUpdate: () => void }

export default function FindingsList({ detections, onUpdate }: Props) {
  // Index duplicates per class name
  const classCount: Record<string, number> = {};
  const indexed = detections.map(d => {
    classCount[d.class_name] = (classCount[d.class_name] ?? 0) + 1;
    return { ...d, _idx: classCount[d.class_name] };
  });
  const multiClass = new Set(
    Object.entries(classCount).filter(([, c]) => c > 1).map(([k]) => k)
  );

  const aiDets   = indexed.filter(d => d.source === "auto");
  const userDets = indexed.filter(d => d.source === "user");

  const remove = async (id: number) => {
    await api.deleteDetection(id);
    onUpdate();
  };

  const Row = (det: typeof indexed[0]) => {
    const conf = CONF_BAR(det.score);
    const label = multiClass.has(det.class_name)
      ? `${det.class_name} #${det._idx}`
      : det.class_name;

    return (
      <div key={det.id} className="group flex items-center gap-2.5 px-3 py-2.5 rounded-xl
        border border-slate-100 bg-white hover:border-slate-200 hover:shadow-sm transition-all">
        {/* Category badge */}
        <span className={`badge shrink-0 text-[10px] px-1.5 py-0.5 ${CAT_COLOR[det.category] ?? CAT_COLOR.Unknown}`}>
          {det.category}
        </span>

        {/* Class + label */}
        <span className="flex-1 font-medium text-sm text-slate-800 truncate" title={label}>
          {label}
        </span>

        {/* Confidence bar + % */}
        <div className="flex items-center gap-1.5 shrink-0">
          <div className="w-12 h-1.5 bg-slate-100 rounded-full overflow-hidden">
            <div className={`h-full rounded-full ${conf.bar}`} style={{ width: `${det.score * 100}%` }} />
          </div>
          <span className={`text-[11px] font-semibold w-7 text-right ${conf.text}`}>
            {(det.score * 100).toFixed(0)}%
          </span>
        </div>

        {/* Remove */}
        <button onClick={() => remove(det.id)}
          title="Remove finding"
          className="shrink-0 w-5 h-5 rounded-full flex items-center justify-center text-slate-300
            hover:text-red-500 hover:bg-red-50 opacity-0 group-hover:opacity-100 transition-all text-xs">
          ✕
        </button>
      </div>
    );
  };

  if (!detections.length)
    return <p className="text-slate-400 text-sm text-center py-6">No findings detected.</p>;

  return (
    <div className="space-y-3">
      {/* AI detections */}
      {aiDets.length > 0 && (
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">🤖 AI Detection</span>
            <span className="text-[10px] text-slate-300">({aiDets.length})</span>
          </div>
          {aiDets.map(Row)}
        </div>
      )}

      {/* User detections */}
      {userDets.length > 0 && (
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">✏️ Operator</span>
            <span className="text-[10px] text-slate-300">({userDets.length})</span>
          </div>
          {userDets.map(Row)}
        </div>
      )}

      {/* Footer */}
      <div className="pt-2 border-t border-slate-50 flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-slate-300" />
        <span className="text-[10px] text-slate-400">
          AI: YOLOv11m fine-tuned on DentexChallenge 2023 · hover a finding to remove
        </span>
      </div>
    </div>
  );
}
