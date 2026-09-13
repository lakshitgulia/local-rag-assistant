#!/usr/bin/env bash
# One-command setup + run for the pitch call.
set -e

cd "$(dirname "$0")"

if ! command -v ollama >/dev/null; then
  echo "Ollama is not installed. Install it from https://ollama.com first." >&2
  exit 1
fi

[ -f .env ] && source .env

for model in "${OLLAMA_EMBED_MODEL:-nomic-embed-text}" "${OLLAMA_LLM_MODEL:-llama3.2:3b}"; do
  if ! ollama list | grep -q "$model"; then
    echo "Pulling missing Ollama model: $model"
    ollama pull "$model"
  fi
done

if [ ! -d venv ]; then
  python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt

if [ ! -f index/chunks.json ]; then
  echo "No index found yet. Running ingestion against \${LOCAL_SCAN_DIR:-~/Downloads}..."
  python ingest.py
fi

echo "Starting server on http://0.0.0.0:8000"
uvicorn app:app --host 0.0.0.0 --port 8000
