#!/bin/bash

# Start FastAPI backend in the background on local interface only
echo "🚀 Starting FastAPI backend on port 8000..."
PYTHONPATH=. python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 &

# Wait for backend to spin up
sleep 3

# Start Streamlit frontend on port specified by PORT env var (defaults to 7860 for Hugging Face)
PORT=${PORT:-7860}
echo "🚀 Starting Streamlit frontend on port $PORT..."
streamlit run src/frontend.py --server.port $PORT --server.address 0.0.0.0
