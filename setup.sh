#!/bin/bash

# Verona AI Search - Setup Script
set -e

echo "🚀 Setting up Verona AI Search..."

# Check if Python 3.11+ is available
PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "❌ Python 3.11+ is required but not found"
    exit 1
fi

echo "📦 Using Python: $PYTHON_CMD"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

# Fix transformers version for FlagEmbedding compatibility
echo "📦 Fixing transformers version..."
pip install "transformers>=4.36.0,<4.46.0" -q

# Set tokenizers parallelism to avoid warnings
export TOKENIZERS_PARALLELISM=false

# Check if .env.development exists
if [ ! -f ".env.development" ]; then
    echo "⚠️  .env.development not found. Creating from .env.example..."
    cp .env.example .env.development
    echo "⚠️  Please edit .env.development and add your OPENAI_API_KEY"
fi

# Check for OpenAI API key
if ! grep -q "OPENAI_API_KEY=sk-" .env.development 2>/dev/null; then
    echo ""
    echo "⚠️  Warning: OPENAI_API_KEY not set in .env.development"
    echo "   Vibe report generation will not work without it."
    echo ""
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the server, run:"
echo "  source venv/bin/activate"
echo "  APP_ENV=development uvicorn app.main:app --reload"
echo ""
echo "Or use the start script:"
echo "  ./start.sh"
