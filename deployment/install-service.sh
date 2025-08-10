#!/bin/bash
# Install MCP Image Server as systemd service

set -e

echo "🔧 Installing MCP Image Server as systemd service..."

# Get current user and paths
CURRENT_USER=$(whoami)
CURRENT_GROUP=$(id -gn)
PROJECT_ROOT=$(realpath ..)
SERVICE_FILE="mcp-image-server.service"

# Create a custom service file with correct paths
cat > "$SERVICE_FILE.tmp" << EOF
[Unit]
Description=MCP Image Generation Server
After=network.target
Wants=network.target

[Service]
Type=simple
User=$CURRENT_USER
Group=$CURRENT_GROUP
WorkingDirectory=$PROJECT_ROOT/mcp_image_server
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=$PROJECT_ROOT/.env
ExecStart=$HOME/.local/bin/uv run python server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=mcp-image-server

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectSystem=strict
ReadWritePaths=$PROJECT_ROOT/data

[Install]
WantedBy=multi-user.target
EOF

echo "📋 Service configuration:"
echo "   User: $CURRENT_USER"
echo "   Project: $PROJECT_ROOT"
echo "   Working Dir: $PROJECT_ROOT/mcp_image_server"

# Install service
echo "📦 Installing service file..."
sudo cp "$SERVICE_FILE.tmp" "/etc/systemd/system/$SERVICE_FILE"
sudo systemctl daemon-reload

# Clean up temp file
rm "$SERVICE_FILE.tmp"

echo "🔄 Enabling service..."
sudo systemctl enable mcp-image-server

echo "✅ Service installed successfully!"
echo ""
echo "📋 Usage:"
echo "   sudo systemctl start mcp-image-server     # Start service"
echo "   sudo systemctl stop mcp-image-server      # Stop service"
echo "   sudo systemctl status mcp-image-server    # Check status"
echo "   sudo journalctl -u mcp-image-server -f    # View logs"
echo ""
echo "🚀 To start now: sudo systemctl start mcp-image-server"