#!/bin/bash

# Start FastAPI backend in the background
echo "🚀 Starting FastAPI backend on port 8000..."
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend in the foreground
echo "🚀 Starting Streamlit frontend on port 8501..."
streamlit run src/frontend.py --server.port 8501 --server.address 0.0.0.0
