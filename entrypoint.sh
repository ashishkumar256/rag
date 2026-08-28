#!/bin/bash
set -e

# Start static PDF server in background
python serve_pdfs.py --host 0.0.0.0 --port ${PDF_PORT:-8080} &
PDF_PID=$!

# Start the RAG API (foreground)
python app.py &
APP_PID=$!

# Wait for either to exit
wait -n $PDF_PID $APP_PID
exit $?
