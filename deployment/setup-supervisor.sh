#!/bin/bash
# Supervisor setup script for MCP Image Server

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USERNAME=$(whoami)
HOME_DIR=$(eval echo ~$USERNAME)

echo "🔧 Setting up Supervisor for MCP Image Server"
echo "=" * 50

# Check if supervisor is installed
if ! command -v supervisord &> /dev/null; then
    echo "❌ Supervisor not found. Installing..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt update && sudo apt install -y supervisor
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install supervisor
    else
        echo "❌ Unsupported OS. Please install supervisor manually."
        exit 1
    fi
fi

# Create custom supervisor config
SUPERVISOR_CONFIG="$PROJECT_ROOT/deployment/supervisor.conf"
cat > "$SUPERVISOR_CONFIG" << EOF
[program:mcp-image-server]
command=$HOME_DIR/.local/bin/uv run python server.py
directory=$PROJECT_ROOT/mcp_image_server
environment=PATH="$HOME_DIR/.local/bin:/usr/local/bin:/usr/bin:/bin"
user=$USERNAME
autostart=true
autorestart=true
startsecs=10
startretries=3
stdout_logfile=$PROJECT_ROOT/logs/supervisor-mcp.log
stderr_logfile=$PROJECT_ROOT/logs/supervisor-mcp-error.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=3
stderr_logfile_maxbytes=10MB
stderr_logfile_backups=3
EOF

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

# Check if .env exists
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "❌ .env file not found!"
    echo "   Please create .env with your REPLICATE_API_TOKEN"
    exit 1
fi

# Create data directories
mkdir -p "$PROJECT_ROOT/data/generated_images" "$PROJECT_ROOT/data/cache"

echo "✅ Configuration created: $SUPERVISOR_CONFIG"
echo ""
echo "🚀 To start the service:"
echo "   supervisord -c $SUPERVISOR_CONFIG"
echo ""
echo "📊 Management commands:"
echo "   supervisorctl -c $SUPERVISOR_CONFIG status"
echo "   supervisorctl -c $SUPERVISOR_CONFIG start mcp-image-server"
echo "   supervisorctl -c $SUPERVISOR_CONFIG stop mcp-image-server"
echo "   supervisorctl -c $SUPERVISOR_CONFIG restart mcp-image-server"
echo "   supervisorctl -c $SUPERVISOR_CONFIG tail -f mcp-image-server"
echo ""
echo "📁 Logs location: $PROJECT_ROOT/logs/"
echo "📁 Data location: $PROJECT_ROOT/data/"

# Make the config file executable
chmod +x "$SUPERVISOR_CONFIG"

echo ""
echo "✅ Supervisor setup complete!"
EOF