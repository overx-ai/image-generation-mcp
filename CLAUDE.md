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
```

### Configuration
- API tokens: Set in `.env` file: `REPLICATE_API_TOKEN=your_token` (optional `OPENAI_API_KEY` for gpt-image-2 billed to your OpenAI account)
- Main config: `mcp_image_server/config/config.yaml` (YAML format only)
- Model definitions: `ReplicateProvider.MODEL_CONFIGS` in `mcp_image_server/providers/replicate.py` (source of truth — `config/models/config.yaml` is unused legacy)

### Self-check
```bash
uv run python -m mcp_image_server.server --demo   # output/reference helpers, fallback chain + per-model inputs, async job lifecycle; no API needed
```

## Architecture Overview

### Core Components

**MCP Server (`server.py`)**
- FastMCP server exposing 10 tools (generation, async jobs, listing, cache, optimization) — see [docs/001](docs/001-image-generation-tools.md)
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

Full reference: [docs/001 - Image Generation Tools](docs/001-image-generation-tools.md).

- **generate_blog_image / generate_premium_image / generate_text_image**: blocking generation (budget / high-quality / text-tuned)
- **start_image_generation / get_image_result**: async start+poll — use when the client has a short tool-call timeout (e.g. Claude Cowork)
- **list_generated_images / get_cache_stats / clear_cache**: inventory + cache management
- **optimize_existing_image / batch_optimize_images**: web optimization of existing images

### Generation conventions
- All `generate_*` (and `start_image_generation`) accept optional `output_dir` + `filename` (custom dir skips `YYYY-MM/DD` subfolders; honored on cache hits) and `reference_image` (local path or URL, single/list; gpt-image-2 + nano-banana only).
- `quality` defaults to `medium` on every tool (~$0.047/image on gpt-image-2) and also selects the fallback tier. Keep `medium` — tool docstrings recommend it; raise to `high` ($0.128) only for dense or very small typography.
- Automatic fallback on failure: gpt-image-2 → same model without our `OPENAI_API_KEY` (Replicate-billed, fixes "sufficient credits (E001)") → `nano-banana` (`google/nano-banana-2` for low/medium, `google/nano-banana-pro` for high/auto) → `bytedance/seedream-3`. Chain lives in `config.yaml` under `models.fallback`; the result carries `fallback_from` when a substitute ran. See [docs/002](docs/002-model-fallback.md).
- Never hardcode per-model Replicate inputs in `server.py` — declare them in `ReplicateProvider.MODEL_CONFIGS` (`supports_megapixels`, `supported_ratios`, `extra_input`, `supports_output_format`, `image_input_param`) and let `build_model_input()` apply them. Read a model's live schema from the Replicate API before adding an entry; models reject inputs they don't declare.
- Client timeouts: Claude Code honors per-server `timeout` in `~/.claude.json` + `MCP_TOOL_TIMEOUT`; **Cowork/Desktop do not** (fixed ~30–60s) → use the async tools there.
- Cache key includes `quality` and `reference_image_id` when set (appended only when present so legacy keys stay stable).

## Testing Strategy

- `--demo` (see Self-check above) is the assert-based check that must stay green; extend
  `demo()` / `_demo_fallback()` / `_demo_async_jobs()` in `server.py` when adding pure logic.
- Real generation is verified ad-hoc against the Replicate API (costs credits) — monkeypatch
  `ReplicateProvider._generate_image` to force a failure when exercising the fallback chain,
  see [docs/002](docs/002-model-fallback.md#verification).

## Configuration Notes

Always use uv for Python dependency management. The project uses pyproject.toml for modern Python packaging with entry points configured for the MCP server.

API tokens should be stored in the .env file or set as environment variables for security. The server supports both Replicate and Eden AI providers through the abstract base class pattern. Configuration uses only YAML format for clarity and consistency.

## Code comments

- **A comment is written only when it is essential to understanding intricate logic** — a non-obvious
  algorithm, a workaround and why the obvious way fails, an invariant the code cannot state. If the
  code already says it, the comment is noise and gets deleted.
- **No boilerplate docstrings.** Nothing on getters, setters, DTOs, config, plumbing, or anything the
  signature already says. Never a comment "just in case".
- Comments rot; code does not. A wrong comment is worse than none.
