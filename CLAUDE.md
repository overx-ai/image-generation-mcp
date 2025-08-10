# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MCP (Model Context Protocol) server that provides AI image generation capabilities for blog posts and content creation. The server uses Replicate API as the backend provider and implements a sophisticated caching system, retry logic, and provider abstraction.

## Development Commands

### Environment Setup
```bash
# Install dependencies and create virtual environment
uv sync

# Run the MCP server
uv run python mcp_image_server/server.py

# Run standalone tests
uv run python mcp_image_server/simple_test.py
uv run python mcp_image_server/test_retry_simple.py
```

### Configuration
- API tokens: Set in `.env` file: `REPLICATE_API_TOKEN=your_token`
- Main config: `mcp_image_server/config.yaml` (YAML format only)
- Model definitions: `mcp_image_server/models/config.yaml`

## Architecture Overview

### Core Components

**MCP Server (`server.py`)**
- FastMCP server exposing 4 tools: `generate_blog_image`, `list_generated_images`, `get_cache_stats`, `clear_cache`
- Handles configuration loading (YAML primary, JSON fallback)
- Integrates cache system and provider architecture
- Comprehensive logging to file and console

**Provider Architecture (`providers/`)**
- `base.py`: Abstract base class with retry logic, cost estimation, and common functionality
- `replicate.py`: Replicate API implementation with model mapping and friendly names
- `eden_ai.py`: Eden AI provider (not fully implemented)
- Providers handle exponential backoff retry with jitter and detailed error logging

**Cache System (`cache.py`)**
- File-based cache with SHA256 keys generated from prompt+model+style+dimensions
- TTL-based expiration (30 days default) and LRU cleanup when size limits exceeded
- Atomic operations with metadata stored in JSON sidecar files

### Key Design Patterns

**Configuration Hierarchy**
1. Environment variables (highest priority)
2. .env file variables
3. YAML config file
4. Hardcoded defaults

**Error Handling Strategy**
- Comprehensive logging at INFO/ERROR levels
- Retry logic with exponential backoff and jitter
- Graceful degradation (cache misses, API failures)
- MCP context integration for user feedback

**File Organization**
- `generated_images/YYYY-MM/DD/` for date-based organization
- `cache/` directory for cached images and metadata
- JSON metadata files alongside all images

## MCP Tools

1. **generate_blog_image**: Core image generation with caching and retry logic
2. **list_generated_images**: Inventory management with metadata
3. **get_cache_stats**: Monitoring cache usage and configuration  
4. **clear_cache**: Maintenance tool for expired entry cleanup

## Testing Strategy

- `simple_test.py`: End-to-end generation test
- `test_retry_simple.py`: Retry logic validation
- `test_*.py`: Various component tests
- Tests use mock tokens and demonstrate proper error handling

## Configuration Notes

Always use uv for Python dependency management. The project uses pyproject.toml for modern Python packaging with entry points configured for the MCP server.

API tokens should be stored in the .env file or set as environment variables for security. The server supports both Replicate and Eden AI providers through the abstract base class pattern. Configuration uses only YAML format for clarity and consistency.