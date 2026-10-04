---
status: current
created: 2026-07-22
updated: 2026-09-03
---

# 001 - Image Generation Tools

**Related**: [000 - Changelog](000-changelog.md), [002 - Model Fallback](002-model-fallback.md)

Reference for the MCP tools exposed by `mcp_image_server/server.py`. Backend is
Replicate; default model is `openai/gpt-image-2` (quality `medium`), with an
automatic fallback chain when it fails — see [002](002-model-fallback.md).
Generated images are saved locally (with a JSON metadata sidecar) and cached by
content.

## Tool inventory

| Tool | Purpose |
|------|---------|
| `generate_blog_image` | Blocking generation, gpt-image-2 at `medium` quality |
| `generate_premium_image` | Blocking generation; FLUX/Ideogram/Recraft/Seedream/nano-banana selectable |
| `generate_text_image` | Blocking generation tuned for readable text/logos |
| `start_image_generation` | **Async**: start a job, return `job_id` immediately |
| `get_image_result` | **Async**: poll a job started by `start_image_generation` |
| `list_generated_images` | Inventory of saved images + metadata |
| `get_cache_stats` | Cache usage/config |
| `clear_cache` | Remove expired cache entries |
| `optimize_existing_image` | Web-optimize one existing image |
| `batch_optimize_images` | Web-optimize a directory |

## Common generation params (all `generate_*` + `start_image_generation`)

| Param | Default | Notes |
|-------|---------|-------|
| `prompt` | — | required |
| `aspect_ratio` | — | required (blog); `16:9` default (text). `landscape/portrait/square` or `1:1,16:9,21:9,3:2,2:3,4:5,5:4,3:4,4:3,9:16,9:21`. gpt-image-2 maps to its nearest supported ratio |
| `style` | `photorealistic` | `photorealistic`, `photorealistic_with_text`, `artistic`, `cartoon`, `sketch` |
| `model` | config default (`openai/gpt-image-2`) | see model list in tool docstrings |
| `quality` | `medium` (all tools, **recommended — leave it alone**; `high` only for dense/small typography) | gpt-image-2 only: `low` $0.012 / `medium` $0.047 / `high` $0.128 / `auto`; ignored by other models. Also picks the fallback tier (see below) |
| `output_dir` | config `storage.base_path` | **When set, saves straight here — no `YYYY-MM/DD` subfolders.** Prefer absolute paths; relative resolves against server cwd |
| `filename` | auto `image_<uuid>.png` | `.png` appended if no image extension given |
| `reference_image` | none | local path **or** http(s) URL, single or list. Supported by `openai/gpt-image-2` and both nano-bananas (others return a clear error) |

**File**: `mcp_image_server/server.py` — helpers `resolve_output_base`,
`generate_local_path`, `resolve_reference_images`, `save_cached_to_output`.

### Automatic fallback

A failing model does not return an error to the client — the server retries
gpt-image-2 without our OpenAI key (Replicate-billed), then switches to a
quality-matched `nano-banana`, then `bytedance/seedream-3`. Results carry
`fallback_from` when a substitute produced the image; `model`/`cost` report what
actually ran.

**Details**: [002 - Model Fallback & Per-Model Inputs](002-model-fallback.md).

### Output location
`resolve_output_base(config, output_dir)` returns `(base, organize_by_date)`.
A custom `output_dir` disables date subfolders; the config default keeps
`storage.base_path/YYYY-MM/DD/`. On a **cache hit** with a custom location, the
cached image is copied there via `save_cached_to_output` (source extension
preserved when `filename` has none).

### Reference images (image-to-image seed)
`resolve_reference_images` validates local paths and passes them as `Path`
objects — the Replicate client uploads them directly (no base64/manual upload).
For gpt-image-2 the value is injected as `input_images` (an array); see
`ReplicateProvider.MODEL_CONFIGS["openai/gpt-image-2"]` (`image_input_param`).
The reference identity (URL, or sha256 of file bytes) is folded into the cache
key so referenced generations don't collide with plain-prompt ones.

## Async generation (`start_image_generation` + `get_image_result`)

Use the async pair when the **client enforces a short tool-call timeout** — most
importantly **Claude Cowork** (~30–60s, not configurable), where a 1–3 min render
would otherwise be killed client-side. The blocking `generate_*` tools are fine
in Claude Code once its timeout is raised (see below).

Flow:
1. `start_image_generation(...)` (same params as `generate_premium_image`) →
   `{status: "processing", job_id, ...}` immediately. It runs the existing
   `generate_premium_image` flow as a background `asyncio.Task` on the server.
2. Poll `get_image_result(job_id)` every ~20–30s →
   `processing` | `completed` (with `image_url`) | `failed` | `unknown`.
   Results are returned **once**, then the job is dropped (a second poll →
   `unknown`).

Job registry `GENERATION_JOBS` is in-memory — jobs are lost if the server
process restarts mid-render (re-trigger to recover).

## Client timeouts

| Client | Behavior | Fix |
|--------|----------|-----|
| Claude Code (CLI) | Reads `~/.claude.json` + `~/.claude/settings.json` | Per-server `"timeout"` on the `image-gen-mcp` entry (set to `600000`) and `MCP_TOOL_TIMEOUT=600000` in settings `env`. Blocking tools then work. Takes effect on session restart |
| Claude Cowork / Desktop | Own config (`claude_desktop_config.json`); tool timeout ~30–60s, not configurable | Use the **async** tools above |

## Caching
`ImageCache.generate_cache_key` (`mcp_image_server/core/cache.py`) hashes
prompt + model + style + aspect/dimensions, plus `quality` and
`reference_image_id` when present (appended only when set, so legacy keys stay
stable). 30-day TTL, LRU cleanup, JSON metadata sidecars.

## Self-check
`uv run python -m mcp_image_server.server --demo` runs assert-based checks for the
output/reference helpers, the fallback chain / per-model inputs and the async job
lifecycle (no API/credit needed).
