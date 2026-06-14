import { useEffect, useState, useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { formatDate } from "../utils/date";
import type { AnalysisListItem } from "../types";

const STATUS_MAP: Record<string, { color: string; dot: string; label: string }> = {
  done:       { color: "bg-emerald-50 text-emerald-700", dot: "bg-emerald-400",              label: "Done" },
  processing: { color: "bg-amber-50 text-amber-700",    dot: "bg-amber-400 animate-pulse",   label: "Processing" },
  enriching:  { color: "bg-blue-50 text-blue-700",      dot: "bg-blue-400 animate-pulse",    label: "Enriching" },
  pending:    { color: "bg-slate-100 text-slate-500",   dot: "bg-slate-300",                 label: "Pending" },
  error:      { color: "bg-red-50 text-red-600",        dot: "bg-red-400",                   label: "Error" },
};

export default function ListPage() {
  const [analyses, setAnalyses] = useState<AnalysisListItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging]   = useState(false);
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    setAnalyses(await api.listAnalyses());
  }, []);

  useEffect(() => {
    const tick = () => { if (!document.hidden) load(); };
    tick();
    const t = setInterval(tick, 3000);
    const onVisibility = () => { if (!document.hidden) load(); };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      clearInterval(t);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [load]);

  const handleFile = async (file: File) => {
    if (!file.type.startsWith("image/")) return;
    setUploading(true);
    try {
      const { id } = await api.upload(file);
      navigate(`/analysis/${id}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-800">Radiographic analyses</h2>
        <p className="text-slate-500 text-sm mt-1">
          Upload a panoramic dental X-ray to automatically detect pathologies and generate a clinical report.
        </p>
      </div>

      {/* Upload zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
        className={`relative cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-200 p-10 text-center select-none
          ${dragging
            ? "border-blue-400 bg-blue-50 scale-[1.01]"
            : "border-slate-300 bg-white hover:border-blue-300 hover:bg-blue-50/30"}`}
      >
        <input ref={inputRef} type="file" accept="image/*" className="hidden"
          onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />

        {uploading ? (
          <div className="flex flex-col items-center gap-3">
            <div className="w-10 h-10 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
            <p className="text-blue-700 font-semibold">Uploading and starting analysis...</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <div className="w-14 h-14 bg-blue-50 rounded-2xl flex items-center justify-center mb-1">
              <svg className="w-7 h-7 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
              </svg>
            </div>
            <p className="font-semibold text-slate-700">Drop an OPG image here or click to select</p>
            <p className="text-slate-400 text-xs">JPEG, PNG — panoramic dental X-ray</p>
          </div>
        )}
      </div>

      {/* Table */}
      {analyses.length === 0 ? (
        <div className="card text-center py-16">
          <p className="text-4xl mb-3">🦷</p>
          <p className="text-slate-500 font-medium">No analyses yet.</p>
          <p className="text-slate-400 text-sm">Upload an image above to get started.</p>
        </div>
      ) : (
        <div className="card p-0 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-100 bg-slate-50/60">
                {["#", "Preview", "File", "Status", "Date", ""].map(h => (
                  <th key={h} className="px-5 py-3 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {analyses.map(a => {
                const s = STATUS_MAP[a.status] ?? STATUS_MAP.pending;
                return (
                  <tr key={a.id} onClick={() => navigate(`/analysis/${a.id}`)}
                    className="hover:bg-slate-50/60 transition-colors cursor-pointer group">
                    <td className="px-5 py-3.5 text-slate-400 font-mono text-xs">{a.id}</td>
                    <td className="px-5 py-3.5">
                      <img src={a.image_url} alt=""
                        className="w-16 h-9 object-cover rounded-lg border border-slate-100 shadow-sm" />
                    </td>
                    <td className="px-5 py-3.5 font-medium text-slate-800">{a.filename}</td>
                    <td className="px-5 py-3.5">
                      <span className={`badge ${s.color} gap-1.5`}>
                        <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${s.dot}`} />
                        {s.label}
                      </span>
                    </td>
                    <td className="px-5 py-3.5 text-slate-400 text-xs">
                      {formatDate(a.created_at)}
                    </td>
                    <td className="px-5 py-3.5">
                      <span className="text-blue-600 font-medium text-xs opacity-0 group-hover:opacity-100 transition-opacity">
                        Open →
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
