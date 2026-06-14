import { useState } from "react";
import VoiceInput from "./VoiceInput";

interface Pending { x1: number; y1: number; x2: number; y2: number; cls: string }
interface Props {
  analysisId: number;
  pendingRegions: Pending[];
  onEnrich: (classNames: string[], contextNotes: string) => void;
  onUpdate: () => void;
  enriching: boolean;
  onClearRegions: () => void;
}

export default function EnrichPanel({ pendingRegions, onEnrich, enriching, onClearRegions }: Props) {
  const [notes, setNotes] = useState("");

  const addTranscript = (t: string) => setNotes(p => p ? `${p} ${t}` : t);
  const canEnrich = notes.trim().length > 0 || pendingRegions.length > 0;

  return (
    <div className="card border-amber-200 bg-amber-50/40 space-y-5">
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 bg-amber-100 rounded-xl flex items-center justify-center text-base shrink-0">✏️</div>
        <div>
          <h3 className="font-semibold text-slate-800 text-sm">Enrich report</h3>
          <p className="text-xs text-slate-500">Add clinical context to generate an enriched pre-report (V2)</p>
        </div>
      </div>

      {/* Notes */}
      <div className="space-y-2">
        <label className="section-title">Clinical notes</label>
        <VoiceInput onTranscript={addTranscript} />
        <textarea value={notes} onChange={e => setNotes(e.target.value)}
          rows={3} placeholder="Describe additional findings or observations..."
          className="!bg-white border-slate-200 focus:border-amber-400 resize-none" />
      </div>

      {/* Pending drawn regions */}
      {pendingRegions.length > 0 && (
        <div className="bg-white rounded-xl border border-amber-200 p-3 space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-amber-700">{pendingRegions.length} region(s) drawn</span>
            <button onClick={onClearRegions} className="text-xs text-red-400 hover:text-red-600 transition-colors">Clear</button>
          </div>
          {pendingRegions.map((r, i) => (
            <div key={i} className="text-xs text-slate-500 flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-amber-400 rounded-full shrink-0" />
              <span className="font-medium text-slate-700">{r.cls}</span>
              <span className="text-slate-400 font-mono text-[10px]">
                ({r.x1.toFixed(0)},{r.y1.toFixed(0)})→({r.x2.toFixed(0)},{r.y2.toFixed(0)})
              </span>
            </div>
          ))}
        </div>
      )}

      <button onClick={() => onEnrich([], notes)} disabled={enriching || !canEnrich}
        className="w-full py-3 rounded-xl font-semibold text-sm transition-all
          bg-amber-600 hover:bg-amber-700 active:bg-amber-800 text-white
          disabled:opacity-40 disabled:cursor-not-allowed shadow-sm">
        {enriching
          ? <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Generating enriched pre-report...
            </span>
          : "Generate enriched pre-report (V2)"}
      </button>
    </div>
  );
}
