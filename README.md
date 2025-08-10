# MCP Image Generation Server

AI-powered image generation server using the Model Context Protocol (MCP) and Replicate API.

## Quick Start

### 1. Setup Environment

```bash
# Clone and install dependencies
uv sync

# Configure API token
cp .env.example .env
# Edit .env and set your REPLICATE_API_TOKEN
```

### 2. Get API Token

1. Go to [Replicate.com](https://replicate.com/)
2. Sign up/login → Account Settings → API Tokens
3. Copy your token to `.env` file

### 3. Run MCP Server

```bash
cd mcp_image_server
uv run python server.py
```

## MCP Tools

The server provides 4 MCP tools:

- **`generate_blog_image`** - Generate images from text prompts
- **`list_generated_images`** - List all generated images  
- **`get_cache_stats`** - View cache statistics
- **`clear_cache`** - Clear expired cache entries

## Usage with MCP Client

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "image-generation": {
      "command": "uv",
      "args": ["run", "python", "server.py"],
      "cwd": "/path/to/image-generation-mcp/mcp_image_server"
    }
  }
}
```

## Configuration

- **Images:** Stored in `data/generated_images/` (outside code directory)
- **Cache:** Stored in `data/cache/` (30-day TTL, 1GB limit)
- **Config:** `mcp_image_server/config/config.yaml`

## Features

- ✨ High-quality image generation (FLUX model)
- 🗄️ Intelligent caching system
- 🔄 Retry logic with exponential backoff
- 📁 Organized file storage by date
- 📊 Complete metadata tracking
- 🏠 External data storage (outside code)

## Deployment Options

### Local Development
```bash
# Quick start script
./start.sh
```

### Docker (Recommended)
```bash
# One-command Docker setup
./docker-run.sh
```
Images accessible in `./data/generated_images/` on host machine.

### Linux Systemd Service
```bash
# Install as system service
sudo ./deployment/install-systemd.sh
```

### Process Manager (Supervisor)
```bash
# Setup supervisor
./deployment/setup-supervisor.sh
# Start service
supervisord -c deployment/supervisor.conf
```

## Generated Images Access

- **Local/Docker**: `./data/generated_images/YYYY-MM/DD/`
- **MCP tool**: Use `list_generated_images` to inventory all images
- **Persistence**: All deployment options maintain data outside containers

Generated images are saved with metadata and cached for efficiency. Cost: ~$0.025 per image.