import { useState, useRef } from "react";

const LANGS = [
  { label: "PT-BR", value: "pt-BR" },
  { label: "English", value: "en-US" },
  { label: "Español", value: "es-ES" },
];

interface Props { onTranscript: (text: string) => void }

export default function VoiceInput({ onTranscript }: Props) {
  const [listening, setListening] = useState(false);
  const [lang, setLang]           = useState("pt-BR");
  const [error, setError]         = useState("");
  const recRef = useRef<any>(null);

  const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

  const toggle = () => {
    if (!SR) { setError("Use Chrome for voice recognition."); return; }
    if (listening) { recRef.current?.stop(); setListening(false); return; }
    const rec = new SR();
    rec.lang = lang;
    rec.continuous = false;
    rec.interimResults = false;
    rec.onresult = (e: any) => { onTranscript(e.results[0][0].transcript); setListening(false); };
    rec.onerror = () => { setListening(false); setError("Recognition error."); };
    rec.onend = () => setListening(false);
    recRef.current = rec;
    rec.start();
    setListening(true);
    setError("");
  };

  return (
    <div className="flex items-center gap-2 flex-wrap">
      {/* Language selector */}
      <select value={lang} onChange={e => setLang(e.target.value)} disabled={listening}
        className="select-styled !w-auto !py-1.5 !text-xs border-slate-200 disabled:opacity-50">
        {LANGS.map(l => <option key={l.value} value={l.value}>{l.label}</option>)}
      </select>

      <button type="button" onClick={toggle}
        className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold border transition-all
          ${listening
            ? "bg-red-500 text-white border-red-500 shadow-md shadow-red-100 animate-pulse"
            : "bg-white text-slate-700 border-slate-200 hover:border-blue-300 hover:text-blue-600 shadow-sm"}`}>
        <span className={`w-2 h-2 rounded-full shrink-0 ${listening ? "bg-white" : "bg-red-400"}`} />
        {listening ? "Recording..." : "Record voice"}
      </button>

      {error && <span className="text-red-500 text-xs">{error}</span>}
    </div>
  );
}
