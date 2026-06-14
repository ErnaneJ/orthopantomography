import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

const BASE = "/api";

interface ClassStat {
  class_name: string; category: string; count: number; prevalence: number;
  avg_confidence: number; min_confidence: number; max_confidence: number;
}
interface CategoryStat { category: string; count: number }
interface SourceStat    { source: string; count: number }
interface ConfBin       { bin: string; count: number }
interface Stats {
  total_analyses: number; done_analyses: number; total_detections: number;
  class_stats: ClassStat[]; category_stats: CategoryStat[]; source_stats: SourceStat[];
  confidence_percentiles: Record<string, number>;
  confidence_histogram: ConfBin[];
  findings_per_analysis: number[];
  findings_summary: { mean: number; min: number; max: number; std: number; total_analyses_with_findings: number };
}

const CAT_HEX: Record<string, string> = {
  Diseases: "#EF4444", Treatments: "#3B82F6", "Tooth Status": "#10B981",
  Anatomy: "#F59E0B", Orthodontics: "#8B5CF6", User: "#EAB308", Unknown: "#94A3B8",
};

function Bar({ value, max, color = "#3B82F6", label }: { value: number; max: number; color?: string; label?: string }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div className="flex items-center gap-2 w-full">
      <div className="flex-1 h-4 bg-slate-100 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct.toFixed(1)}%`, background: color }} />
      </div>
      <span className="text-xs font-mono text-slate-500 w-8 text-right">{label ?? value}</span>
    </div>
  );
}

function ConfidenceBar({ value }: { value: number }) {
  const color = value >= 0.80 ? "#16A34A" : value >= 0.60 ? "#D97706" : "#DC2626";
  return (
    <div className="flex items-center gap-2">
      <div className="w-24 h-2.5 bg-slate-100 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${(value * 100).toFixed(0)}%`, background: color }} />
      </div>
      <span className="text-xs font-mono" style={{ color }}>{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

function StatCard({ label, value, sub, color = "text-blue-600" }: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div className="card text-center py-4">
      <p className={`text-3xl font-bold ${color}`}>{value}</p>
      <p className="text-xs font-semibold text-slate-500 mt-1 uppercase tracking-wide">{label}</p>
      {sub && <p className="text-[10px] text-slate-400 mt-0.5">{sub}</p>}
    </div>
  );
}

export default function StatsPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetch(`${BASE}/analyses/stats`)
      .then(r => r.json())
      .then(setStats)
      .catch(e => setError(String(e)));
  }, []);

  if (error)
    return (
      <div className="card border-red-200 bg-red-50 text-red-700 py-8 text-center">
        <p className="font-semibold">Erro ao carregar estatísticas</p>
        <p className="text-xs mt-1 font-mono">{error}</p>
      </div>
    );

  if (!stats)
    return (
      <div className="flex items-center justify-center py-32 gap-3">
        <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin" />
        <span className="text-slate-400 text-sm">Carregando métricas...</span>
      </div>
    );

  const maxCount = stats.class_stats[0]?.count ?? 1;
  const totalCatDets = stats.category_stats.reduce((s, c) => s + c.count, 0);

  // Findings histogram (bin counts)
  const histBins: Record<number, number> = {};
  for (const n of stats.findings_per_analysis) {
    histBins[n] = (histBins[n] ?? 0) + 1;
  }
  const maxHistCount = Math.max(...Object.values(histBins), 1);

  const aiCount  = stats.source_stats.find(s => s.source === "auto")?.count ?? 0;
  const usrCount = stats.source_stats.find(s => s.source === "user")?.count ?? 0;

  return (
    <div className="space-y-6">

      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <button onClick={() => navigate("/")}
            className="text-xs text-slate-400 hover:text-blue-600 flex items-center gap-1 mb-1 transition-colors">
            ← Todas as análises
          </button>
          <h2 className="text-2xl font-bold text-slate-800">Métricas do Dataset</h2>
          <p className="text-slate-500 text-sm mt-1">
            Estatísticas agregadas para suporte a artigo científico
          </p>
        </div>
      </div>

      {/* Overview cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <StatCard label="Análises totais" value={stats.total_analyses}
          sub={`${stats.done_analyses} concluídas`} color="text-slate-700" />
        <StatCard label="Achados válidos" value={stats.total_detections}
          sub={`IA: ${aiCount} · Oper.: ${usrCount}`} color="text-blue-600" />
        <StatCard label="Classes únicas" value={stats.class_stats.length}
          sub="em todos os exames" color="text-purple-600" />
        <StatCard label="Média por exame" value={stats.findings_summary.mean}
          sub={`min ${stats.findings_summary.min} · max ${stats.findings_summary.max} · σ ${stats.findings_summary.std}`}
          color="text-emerald-600" />
      </div>

      {/* Two-column main grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">

        {/* Class frequency */}
        <div className="card">
          <p className="section-title">Frequência de Classes</p>
          <p className="text-xs text-slate-400 mb-4">
            Número de detecções válidas por classe (IA + operador). N = {stats.total_detections}
          </p>
          <div className="space-y-2 max-h-[480px] overflow-y-auto pr-1">
            {stats.class_stats.map(c => (
              <div key={c.class_name} className="grid items-center gap-x-2" style={{ gridTemplateColumns: "140px 1fr 30px" }}>
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ background: CAT_HEX[c.category] ?? "#94A3B8" }} />
                  <span className="text-xs text-slate-700 truncate">{c.class_name}</span>
                </div>
                <Bar value={c.count} max={maxCount} color={CAT_HEX[c.category] ?? "#94A3B8"} />
                <span className="text-xs font-mono text-slate-400 text-right">{c.count}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Confidence per class */}
        <div className="card">
          <p className="section-title">Confiança Média por Classe</p>
          <p className="text-xs text-slate-400 mb-4">
            Score médio de confiança do modelo (verde ≥ 80%, laranja ≥ 60%, vermelho &lt; 60%)
          </p>
          <div className="space-y-2.5 max-h-[480px] overflow-y-auto pr-1">
            {stats.class_stats.map(c => (
              <div key={c.class_name} className="grid items-center gap-x-3"
                style={{ gridTemplateColumns: "140px 1fr" }}>
                <span className="text-xs text-slate-600 truncate">{c.class_name}</span>
                <ConfidenceBar value={c.avg_confidence} />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Second row */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">

        {/* Category distribution */}
        <div className="card">
          <p className="section-title">Distribuição por Categoria</p>
          <p className="text-xs text-slate-400 mb-4">Total: {totalCatDets} achados</p>

          {/* Stacked bar */}
          <div className="h-7 flex rounded-lg overflow-hidden mb-4 shadow-inner">
            {stats.category_stats.map(c => (
              <div key={c.category}
                style={{
                  width: `${(c.count / totalCatDets * 100).toFixed(1)}%`,
                  background: CAT_HEX[c.category] ?? "#94A3B8",
                }}
                title={`${c.category}: ${c.count} (${(c.count / totalCatDets * 100).toFixed(1)}%)`}
              />
            ))}
          </div>

          <div className="space-y-2.5">
            {stats.category_stats.map(c => (
              <div key={c.category} className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-sm shrink-0"
                  style={{ background: CAT_HEX[c.category] ?? "#94A3B8" }} />
                <span className="text-xs text-slate-700 flex-1">{c.category}</span>
                <span className="text-xs font-mono text-slate-500">{c.count}</span>
                <span className="text-xs text-slate-400">
                  {(c.count / totalCatDets * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Confidence histogram */}
        <div className="card">
          <p className="section-title">Histograma de Confiança</p>
          <p className="text-xs text-slate-400 mb-4">
            Distribuição dos scores de confiança (N = {stats.total_detections})
          </p>
          <div className="flex items-end gap-1 h-32">
            {stats.confidence_histogram.map((bin, i) => {
              const maxBin = Math.max(...stats.confidence_histogram.map(b => b.count), 1);
              const heightPct = (bin.count / maxBin) * 100;
              const color = i >= 8 ? "#16A34A" : i >= 6 ? "#65A30D" : i >= 4 ? "#D97706" : "#DC2626";
              return (
                <div key={bin.bin} className="flex-1 flex flex-col items-center gap-0.5">
                  <span className="text-[8px] text-slate-400 font-mono">{bin.count}</span>
                  <div className="w-full rounded-t-sm transition-all"
                    style={{ height: `${heightPct}%`, background: color, minHeight: bin.count > 0 ? "4px" : "0" }} />
                </div>
              );
            })}
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-[9px] text-slate-400">0%</span>
            <span className="text-[9px] text-slate-400">50%</span>
            <span className="text-[9px] text-slate-400">100%</span>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2 text-[10px] text-slate-500">
            <div>P25: <span className="font-mono font-semibold text-slate-700">{(stats.confidence_percentiles.p25 * 100).toFixed(1)}%</span></div>
            <div>P50: <span className="font-mono font-semibold text-slate-700">{(stats.confidence_percentiles.p50 * 100).toFixed(1)}%</span></div>
            <div>P75: <span className="font-mono font-semibold text-slate-700">{(stats.confidence_percentiles.p75 * 100).toFixed(1)}%</span></div>
            <div>P90: <span className="font-mono font-semibold text-slate-700">{(stats.confidence_percentiles.p90 * 100).toFixed(1)}%</span></div>
            <div>Média: <span className="font-mono font-semibold text-blue-600">{(stats.confidence_percentiles.mean * 100).toFixed(1)}%</span></div>
          </div>
        </div>

        {/* Findings per analysis */}
        <div className="card">
          <p className="section-title">Achados por Exame</p>
          <p className="text-xs text-slate-400 mb-4">
            Distribuição de achados por imagem ({stats.findings_summary.total_analyses_with_findings} exames com achados)
          </p>
          <div className="space-y-2">
            {Object.entries(histBins).sort((a, b) => Number(a[0]) - Number(b[0])).map(([n, cnt]) => (
              <div key={n} className="grid items-center gap-x-2" style={{ gridTemplateColumns: "60px 1fr 24px" }}>
                <span className="text-xs text-slate-500 text-right">{n} achado{Number(n) !== 1 ? "s" : ""}</span>
                <Bar value={cnt} max={maxHistCount} color="#8B5CF6" />
                <span className="text-xs font-mono text-slate-400 text-right">{cnt}</span>
              </div>
            ))}
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 grid grid-cols-2 gap-2 text-[10px] text-slate-500">
            <div>Média: <span className="font-mono font-semibold text-slate-700">{stats.findings_summary.mean}</span></div>
            <div>Desvio: <span className="font-mono font-semibold text-slate-700">±{stats.findings_summary.std}</span></div>
            <div>Mínimo: <span className="font-mono font-semibold text-slate-700">{stats.findings_summary.min}</span></div>
            <div>Máximo: <span className="font-mono font-semibold text-slate-700">{stats.findings_summary.max}</span></div>
          </div>
        </div>
      </div>

      {/* Full class table */}
      <div className="card">
        <p className="section-title">Tabela Completa de Classes</p>
        <p className="text-xs text-slate-400 mb-4">
          Estatísticas detalhadas para cada classe detectada.
          "Prevalência" = número de exames com pelo menos 1 detecção desta classe.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-slate-100 bg-slate-50">
                {["Classe", "Categoria", "Detecções", "Prevalência", "Conf. Média", "Conf. Mín.", "Conf. Máx."].map(h => (
                  <th key={h} className="px-4 py-2.5 text-left text-[10px] font-semibold text-slate-400 uppercase tracking-wider whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {stats.class_stats.map((c, i) => (
                <tr key={c.class_name} className={i % 2 === 0 ? "bg-white" : "bg-slate-50/40"}>
                  <td className="px-4 py-2 font-medium text-slate-800">{c.class_name}</td>
                  <td className="px-4 py-2">
                    <span className="inline-flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full" style={{ background: CAT_HEX[c.category] }} />
                      <span className="text-slate-600">{c.category}</span>
                    </span>
                  </td>
                  <td className="px-4 py-2 font-mono text-blue-700 font-semibold">{c.count}</td>
                  <td className="px-4 py-2 font-mono text-slate-600">
                    {c.prevalence} exame{c.prevalence !== 1 ? "s" : ""}
                    {stats.done_analyses > 0 && (
                      <span className="text-slate-400 ml-1">
                        ({(c.prevalence / stats.done_analyses * 100).toFixed(0)}%)
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2">
                    <ConfidenceBar value={c.avg_confidence} />
                  </td>
                  <td className="px-4 py-2 font-mono text-slate-400">{(c.min_confidence * 100).toFixed(1)}%</td>
                  <td className="px-4 py-2 font-mono text-slate-400">{(c.max_confidence * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Methodology note */}
      <div className="card bg-blue-50/50 border-blue-100">
        <p className="section-title text-blue-600">Nota Metodológica</p>
        <div className="text-xs text-slate-600 space-y-1.5 leading-5">
          <p>
            <strong>Modelo de detecção:</strong> ViT-based (Vision Transformer) com fine-tuning para 31 classes de patologias e estruturas dentárias.
            Inferência via pipeline de 3 estágios: detecção → validação → geração de laudo.
          </p>
          <p>
            <strong>Classes:</strong> 5 categorias — Diseases, Treatments, Tooth Status, Anatomy, Orthodontics —
            cobrindo 31 entidades clínicas conforme nomenclatura FDI.
          </p>
          <p>
            <strong>Confiança:</strong> Score de 0–1 do classificador. Limiar de exibição padrão: 0.30.
            Verde ≥ 80% · Laranja ≥ 60% · Vermelho &lt; 60%.
          </p>
          <p>
            <strong>Achados por exame:</strong> Média de {stats.findings_summary.mean} ± {stats.findings_summary.std}
            (intervalo {stats.findings_summary.min}–{stats.findings_summary.max})
            em {stats.findings_summary.total_analyses_with_findings} exames com detecções.
          </p>
        </div>
      </div>
    </div>
  );
}
