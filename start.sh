#!/bin/bash

# Verona AI Search - Start Script
set -e

# Change to script directory
cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export APP_ENV=${APP_ENV:-development}
export TOKENIZERS_PARALLELISM=false

echo "🚀 Starting Verona AI Search (APP_ENV=$APP_ENV)..."
echo ""

# Start uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 3000
