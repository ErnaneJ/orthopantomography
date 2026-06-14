#!/bin/bash
# Inicia o backend e o frontend da aplicação OPG Analysis

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/web/backend/.env"

# Mata processos antigos nas portas 8000 e 5173
echo "🔄 Encerrando processos anteriores..."
lsof -ti :8000 | xargs kill -9 2>/dev/null || true
lsof -ti :5173 | xargs kill -9 2>/dev/null || true
sleep 1

# Carrega .env se a variável não estiver no shell
if [ -z "$OPENROUTER_API_KEY" ] && [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
  echo "❌ OPENROUTER_API_KEY não encontrada."
  echo "   Adicione em: $ENV_FILE"
  echo "   Formato:     OPENROUTER_API_KEY=sk-or-v1-..."
  exit 1
fi

echo "🦷 OPG Analysis — iniciando..."
echo ""

# Backend
echo "▶ Backend  → http://localhost:8000"
cd "$ROOT/web/backend"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

sleep 2

# Frontend
echo "▶ Frontend → http://localhost:5173"
cd "$ROOT/web/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Pronto! Acesse: http://localhost:5173"
echo "   (aguarde ~20s para o modelo carregar)"
echo "   Ctrl+C para parar tudo."
echo ""

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Parado.'" INT TERM
wait
