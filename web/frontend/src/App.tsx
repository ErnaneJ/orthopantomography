import { Routes, Route, useLocation, Link } from 'react-router-dom'
import ListPage from './pages/ListPage'
import DetailPage from './pages/DetailPage'
import StatsPage from './pages/StatsPage'

export default function App() {
  const loc = useLocation()
  const isDetail = loc.pathname.startsWith('/analysis/')
  const isStats  = loc.pathname === '/stats'

  return (
    <div className="min-h-screen bg-slate-100">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
        <div className="max-w-screen-xl mx-auto px-6 h-14 flex items-center gap-3">
          <Link to="/" className="flex items-center gap-2 group">
            <span className="text-xl">🦷</span>
            <span className="font-bold text-slate-800 text-base tracking-tight group-hover:text-blue-700 transition-colors">
              OPG Analysis
            </span>
          </Link>
          <span className="ml-1 text-xs bg-emerald-100 text-emerald-700 font-semibold px-2 py-0.5 rounded-full">YOLOv11m</span>

          <nav className="ml-4 flex items-center gap-1">
            <Link to="/"
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors
                ${!isDetail && !isStats ? "bg-blue-50 text-blue-700" : "text-slate-500 hover:text-slate-800 hover:bg-slate-100"}`}>
              Analyses
            </Link>
            <Link to="/stats"
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors
                ${isStats ? "bg-blue-50 text-blue-700" : "text-slate-500 hover:text-slate-800 hover:bg-slate-100"}`}>
              Metrics
            </Link>
          </nav>

          {isDetail && (
            <span className="ml-auto text-xs text-slate-400">
              YOLOv11m · DentexChallenge · Gemini 2.5 Flash
            </span>
          )}
        </div>
      </header>

      <main className="max-w-screen-xl mx-auto px-6 py-6">
        <Routes>
          <Route path="/" element={<ListPage />} />
          <Route path="/analysis/:id" element={<DetailPage />} />
          <Route path="/stats" element={<StatsPage />} />
        </Routes>
      </main>
    </div>
  )
}
