#!/bin/bash
# MCP Image Server Startup Script

set -e

echo "🚀 Starting MCP Image Server..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and set your REPLICATE_API_TOKEN"
    exit 1
fi

# Check if API token is set
if grep -q "your_replicate_api_token_here" .env; then
    echo "❌ Please set your REPLICATE_API_TOKEN in .env file"
    exit 1
fi

# Change to server directory
cd mcp_image_server

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Please install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install dependencies if needed
echo "📦 Installing dependencies..."
uv sync --quiet

# Create data directories
mkdir -p ../data/generated_images ../data/cache

echo "✅ Starting server on STDIO..."
echo "📁 Images: $(realpath ../data/generated_images)"
echo "💾 Cache: $(realpath ../data/cache)"
echo "🔌 Connect via MCP client or press Ctrl+C to stop"
echo ""

# Start the server
exec uv run python server.py