---
status: current
created: 2026-07-22
updated: 2026-09-03
---

# Documentation Index

MCP image generation server (single Python project). Backend: Replicate.

## Quick Links
- [001 - Image Generation Tools](001-image-generation-tools.md) — tool reference: params, output paths, reference images, async generation, client timeouts, caching
- [002 - Model Fallback & Per-Model Inputs](002-model-fallback.md) — what happens when a model fails, how each model's Replicate input is built, adding a model
- [000 - Changelog](000-changelog.md) — recent changes

## Documentation Tree
```
000-changelog.md
└── 001-image-generation-tools.md
    └── 002-model-fallback.md
```

## See also
- Repo `CLAUDE.md` — architecture overview + conventions for Claude Code
- `uv run python -m mcp_image_server.server --demo` — helpers, fallback chain and async self-check

---
*Last updated: 2026-09-03*
