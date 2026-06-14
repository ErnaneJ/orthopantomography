import type { AnalysisListItem, AnalysisDetail, Annotation } from "../types";

const BASE = "/api";
const json = async (r: Response) => {
  if (r.status === 404) throw new Error("404");
  return r.json();
};

export const api = {
  listAnalyses: (): Promise<AnalysisListItem[]>   => fetch(`${BASE}/analyses`).then(json),
  getAnalysis:  (id: number): Promise<AnalysisDetail> => fetch(`${BASE}/analyses/${id}`).then(json),

  upload: (file: File): Promise<{ id: number; status: string }> => {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${BASE}/analyses/upload`, { method: "POST", body: form }).then(json);
  },

  deleteDetection: (id: number) =>
    fetch(`${BASE}/detections/${id}`, { method: "DELETE" }).then(json),

  detectClass: (analysisId: number, className: string) =>
    fetch(`${BASE}/analyses/${analysisId}/detect-class`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ class_name: className }),
    }).then(json),

  addAnnotation: (analysisId: number, body: object): Promise<Annotation> =>
    fetch(`${BASE}/analyses/${analysisId}/annotations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(json),

  deleteAnnotation: (analysisId: number, annId: number) =>
    fetch(`${BASE}/analyses/${analysisId}/annotations/${annId}`, { method: "DELETE" }).then(json),

  enrich: (analysisId: number, classNames: string[], contextNotes: string) =>
    fetch(`${BASE}/analyses/${analysisId}/enrich`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ class_names: classNames, context_notes: contextNotes }),
    }).then(json),

  pdfUrl: (analysisId: number) => `${BASE}/analyses/${analysisId}/pdf`,
};
