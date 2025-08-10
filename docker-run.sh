#!/bin/bash
# Docker deployment script for MCP Image Server

set -e

echo "🐳 MCP Image Server - Docker Deployment"
echo "=" * 40

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "   Please create .env with your REPLICATE_API_TOKEN"
    exit 1
fi

# Create directories
mkdir -p data/generated_images data/cache logs

echo "📦 Building Docker image..."
docker-compose build

echo "🚀 Starting MCP Image Server container..."
docker-compose up -d

echo "⏳ Waiting for container to start..."
sleep 5

# Check container status
if docker-compose ps | grep -q "Up"; then
    echo "✅ Container started successfully!"
    echo ""
    echo "📋 Container Status:"
    docker-compose ps
    echo ""
    echo "📊 Container Logs (last 10 lines):"
    docker-compose logs --tail=10
    echo ""
    echo "🔧 Management Commands:"
    echo "   docker-compose logs -f        # View live logs"
    echo "   docker-compose restart        # Restart container"
    echo "   docker-compose stop          # Stop container"
    echo "   docker-compose down          # Stop and remove container"
    echo ""
    echo "📁 Data stored in: $(pwd)/data/"
else
    echo "❌ Container failed to start!"
    echo "📊 Logs:"
    docker-compose logs
    exit 1
fi