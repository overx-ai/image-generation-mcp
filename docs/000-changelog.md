---
status: current
created: 2026-07-22
updated: 2026-10-04
---

# 000 - Changelog

**Related**: [001 - Image Generation Tools](001-image-generation-tools.md),
[002 - Model Fallback](002-model-fallback.md)

## [Unreleased]

### Added
- `openai/gpt-image-2.5-flare` (new default for every tool and style) and
  `openai/gpt-image-2.5-sunburst`, called directly on the OpenAI Images API by
  `OpenAIImagesProvider` (Replicate does not host 2.5). References go through
  `/images/edits`. gpt-image-2 (Replicate-billed) is now the first fallback.

### Changed
- Fallback results are no longer cached under the requested model's key, so one
  failed request does not pin a prompt to the substitute model for the cache TTL.

### Added (earlier)
- `output_dir` + `filename` params on all `generate_*` tools — save images to a
  chosen location (custom dir skips `YYYY-MM/DD` subfolders; honored on cache
  hits too).
- `reference_image` param (local path or URL, single/list) for image-to-image
  seeding. Wired for `openai/gpt-image-2` (`input_images`); other models error
  clearly. Folded into the cache key.
- Async generation tools `start_image_generation` + `get_image_result` — start a
  job and poll for the result, so slow renders survive short client tool-call
  timeouts (e.g. Claude Cowork's non-configurable ~30–60s limit).
- `--demo` self-check in `server.py` covering the output/reference helpers, the
  async job lifecycle and the fallback chain (no API needed).
- Automatic model fallback (see [002](002-model-fallback.md)): gpt-image-2 → same model without our `OPENAI_API_KEY`
  (Replicate-billed) → quality-matched `nano-banana` (`google/nano-banana-2` for
  low/medium, `google/nano-banana-pro` for high/auto) → `bytedance/seedream-3`.
  Chain configured in `config.yaml` `models.fallback`; results carry
  `fallback_from` when a substitute produced the image.
- `google/nano-banana-2` + `google/nano-banana-pro` models (native aspect ratios,
  2K, reference-image support, png output).

### Changed
- `ImageCache.generate_cache_key` now takes an optional `reference_image_id`
  (appended only when present; legacy keys unchanged).
- Default `quality` is now `medium` on every generation tool (was `low` on blog,
  `high` on premium/text/async).
- Tool docstrings now consistently recommend `medium` to calling agents:
  `generate_text_image` no longer claims to default to `high`, and
  `start_image_generation` states the default it was silently missing.

### Fixed
- `megapixels: "1"` was sent to every native-aspect-ratio model; it is a FLUX-only
  input and is now gated on `supports_megapixels` (recraft-v3, seedream-3 and
  ideogram-v3-turbo no longer receive it).
- Aspect ratios are clamped per model (`clamp_aspect_ratio`) — models disagree on
  their enums (seedream-3 has no `4:5`, nano-banana no `9:21`), so a fallback no
  longer fails on a ratio the primary model accepted.
- Billing/auth errors (`sufficient credits (E001)`, `API key is invalid (E002)`)
  no longer burn the full retry backoff before the fallback starts.

### Config (operator action, not code)
- Raised Claude Code's MCP tool timeout for `image-gen-mcp`: per-server
  `"timeout": 600000` in `~/.claude.json` and `MCP_TOOL_TIMEOUT=600000` in
  `~/.claude/settings.json`. Does **not** apply to Cowork/Desktop — use the async
  tools there.
