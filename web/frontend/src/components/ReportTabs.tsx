import { useState } from "react";
import ReactMarkdown from "react-markdown";
import type { Report } from "../types";
import { formatDate } from "../utils/date";

// Insert line breaks before each tooth entry so teeth render one per line
function preprocessReport(text: string): string {
  if (!text) return "";
  // Force each "**Tooth N:**" onto its own paragraph
  return text
    .replace(/\.\s+(\*\*Tooth\s+\d+)/g, ".\n\n$1")
    .replace(/\n(\*\*Tooth\s+\d+)/g, "\n\n$1");
}

interface ModalProps { content: string; title: string; onClose: () => void }

function ReportModal({ content, title, onClose }: ModalProps) {
  return (
    <div className="fixed inset-0 z-[100] bg-black/70 flex items-start justify-center p-4 overflow-y-auto"
      onClick={onClose}>
      <div className="bg-white w-full max-w-3xl rounded-2xl shadow-2xl my-8"
        onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between px-8 py-5 border-b border-slate-100">
          <h2 className="font-bold text-slate-800 text-base">{title}</h2>
          <button onClick={onClose}
            className="w-8 h-8 rounded-full bg-slate-100 hover:bg-slate-200 flex items-center justify-center text-slate-500 transition-colors">
            ✕
          </button>
        </div>
        {/* A4-like reading area */}
        <div className="px-12 py-8 text-sm text-slate-700 leading-7
          [&_h2]:text-base [&_h2]:font-bold [&_h2]:text-slate-900 [&_h2]:mt-8 [&_h2]:mb-3
          [&_h2]:border-b [&_h2]:border-slate-200 [&_h2]:pb-2
          [&_h3]:text-sm [&_h3]:font-semibold [&_h3]:text-slate-700 [&_h3]:mt-5 [&_h3]:mb-2
          [&_p]:my-3 [&_p]:text-slate-600 [&_p]:leading-7
          [&_li]:my-2 [&_li]:text-slate-600 [&_li]:leading-6
          [&_ul]:my-3 [&_ul]:list-disc [&_ul]:pl-5
          [&_strong]:text-slate-800 [&_strong]:font-semibold">
          <ReactMarkdown>{preprocessReport(content)}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

interface Props { reports: Report[] }

export default function ReportTabs({ reports }: Props) {
  const [active, setActive]   = useState(1);
  const [expanded, setExpanded] = useState(false);
  const v1 = reports.find(r => r.version === 1);
  const v2 = reports.find(r => r.version === 2);
  const current = active === 1 ? v1 : v2;

  if (!v1)
    return (
      <div className="flex items-center gap-2 text-slate-400 py-4 text-sm">
        <div className="w-4 h-4 border-2 border-slate-200 border-t-blue-400 rounded-full animate-spin" />
        Generating report...
      </div>
    );

  return (
    <>
      {expanded && current && (
        <ReportModal
          content={current.content}
          title={active === 1 ? "Auto-Generated Report" : "Enriched Report"}
          onClose={() => setExpanded(false)}
        />
      )}

      {/* Tabs */}
      <div className="flex gap-1 mb-3 bg-slate-100 p-1 rounded-xl">
        {[
          { v: 1, label: "Auto",   icon: "🤖", exists: !!v1 },
          { v: 2, label: "Enrich", icon: "✏️", exists: !!v2 },
        ].map(({ v, label, icon, exists }) => (
          <button key={v} disabled={!exists} onClick={() => setActive(v)}
            className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg
              text-xs font-semibold transition-all whitespace-nowrap
              ${active === v && exists ? "bg-white text-slate-800 shadow-sm"
                : exists ? "text-slate-400 hover:text-slate-600 cursor-pointer"
                : "text-slate-300 cursor-not-allowed"}`}>
            {icon} {label}
            {!exists && v === 2 && <span className="ml-1 text-[9px] bg-slate-200 text-slate-400 px-1 py-0.5 rounded">—</span>}
          </button>
        ))}
      </div>

      {/* Compact preview — max 6 lines, expandable */}
      {current && (
        <div>
          <div className="relative">
            <div className="text-sm leading-6 text-slate-600 overflow-hidden max-h-48
              [&_h2]:text-sm [&_h2]:font-bold [&_h2]:text-slate-800 [&_h2]:mt-3 [&_h2]:mb-1
              [&_h3]:text-xs [&_h3]:font-semibold [&_h3]:text-slate-600 [&_h3]:mt-2
              [&_p]:my-1.5 [&_p]:leading-5
              [&_li]:my-0.5 [&_li]:leading-5
              [&_ul]:my-1 [&_ul]:list-disc [&_ul]:pl-4
              [&_strong]:text-slate-800 [&_strong]:font-semibold">
              <ReactMarkdown>{preprocessReport(current.content)}</ReactMarkdown>
            </div>
            {/* Fade gradient */}
            <div className="absolute bottom-0 inset-x-0 h-12 bg-gradient-to-t from-white to-transparent pointer-events-none" />
          </div>

          <button onClick={() => setExpanded(true)}
            className="mt-2 w-full py-2 text-xs font-semibold text-blue-600 bg-blue-50
              hover:bg-blue-100 rounded-lg transition-colors flex items-center justify-center gap-1.5">
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
            </svg>
            Read full report
          </button>
        </div>
      )}

      {/* Footer */}
      <div className="mt-3 pt-2 border-t border-slate-100 flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
        <span className="text-[10px] text-slate-400">
          {current?.model} · {current && formatDate(current.created_at)}
        </span>
      </div>
    </>
  );
}
